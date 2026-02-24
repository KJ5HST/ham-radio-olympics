"""
Sync logic for fetching QSOs from QRZ and updating scores.
"""

import binascii
import httpx
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Set, List, Tuple, Dict

from cryptography.fernet import InvalidToken
from database import get_db

logger = logging.getLogger(__name__)
from crypto import decrypt_api_key
from qrz_client import fetch_qsos, QSOData, QRZAPIError
from lotw_client import fetch_lotw_qsos, LoTWError
from grid_distance import grid_distance
from scoring import (
    recompute_match_medals, compute_team_standings,
    compute_triathlon_leaders, get_honorable_mentions, compute_mode_records
)
from dxcc import get_continent_from_callsign
import json


def precompute_records_cache():
    """
    Pre-compute expensive records data and cache in settings table.

    This runs during sync to avoid expensive on-the-fly computation
    when the /records page is loaded.

    Caches:
    - triathlon_leaders: Top triathlon QSOs
    - honorable_mentions: Best QSOs outside competition
    - distance_records: Mode-specific distance records
    - cool_factor_records: Mode-specific cool factor records
    """
    from database import set_setting, get_setting

    logger.info("Pre-computing records cache...")

    # Compute triathlon leaders and notify Discord if standings changed
    try:
        # Load previous podium for comparison
        old_triathlon_json = get_setting("cache_triathlon_leaders", "[]")
        try:
            old_triathlon = json.loads(old_triathlon_json)
        except (json.JSONDecodeError, TypeError):
            old_triathlon = []

        triathlon = compute_triathlon_leaders(limit=3)
        set_setting("cache_triathlon_leaders", json.dumps(triathlon))
        logger.info(f"Cached {len(triathlon)} triathlon leaders")

        # Check if podium changed (different callsigns or positions — ignore score fluctuations)
        old_podium = [l.get("callsign") for l in old_triathlon]
        new_podium = [l.get("callsign") for l in triathlon]
        if triathlon and new_podium != old_podium:
            try:
                from notifications import discord_notify_triathlon_standings
                discord_notify_triathlon_standings(triathlon)
                logger.info("Sent Discord triathlon standings update")
            except Exception as e:
                logger.error(f"Failed to send Discord triathlon notification: {e}")
    except Exception as e:
        logger.error(f"Error computing triathlon leaders: {e}")
        set_setting("cache_triathlon_leaders", json.dumps([]))

    # Compute honorable mentions
    try:
        mentions = get_honorable_mentions()
        set_setting("cache_honorable_mentions", json.dumps(mentions))
        logger.info(f"Cached honorable mentions: {mentions}")
    except Exception as e:
        logger.error(f"Error computing honorable mentions: {e}")
        set_setting("cache_honorable_mentions", json.dumps({
            'longest_distance': None,
            'highest_cool_factor': None,
        }))

    # Compute mode records
    try:
        distance_recs, cf_recs = compute_mode_records()
        set_setting("cache_distance_records", json.dumps(distance_recs))
        set_setting("cache_cool_factor_records", json.dumps(cf_recs))
        logger.info(f"Cached {len(distance_recs)} distance records, {len(cf_recs)} cool factor records")
    except Exception as e:
        logger.error(f"Error computing mode records: {e}")
        set_setting("cache_distance_records", json.dumps([]))
        set_setting("cache_cool_factor_records", json.dumps([]))

    logger.info("Records cache pre-computation complete")


def delete_competitor_qsos(callsign: str) -> int:
    """
    Delete all QSOs and medals for a competitor.

    This prepares for a full reload from QRZ/LoTW by clearing:
    - All QSOs for the competitor
    - All medals for the competitor
    - The last_sync_at timestamp (forces full reload)

    Args:
        callsign: Competitor callsign

    Returns:
        Count of deleted QSOs
    """
    with get_db() as conn:
        # Get count before deletion
        count = conn.execute(
            "SELECT COUNT(*) FROM qsos WHERE competitor_callsign = ?",
            (callsign.upper(),)
        ).fetchone()[0]

        # Delete medals first (references qsos via foreign key relationship)
        conn.execute("DELETE FROM medals WHERE callsign = ?", (callsign.upper(),))

        # Delete QSOs
        conn.execute("DELETE FROM qsos WHERE competitor_callsign = ?", (callsign.upper(),))

        # Clear last_sync_at to force full reload
        conn.execute(
            "UPDATE competitors SET last_sync_at = NULL WHERE callsign = ?",
            (callsign.upper(),)
        )
        conn.commit()

    # Recompute records from scratch to remove orphaned qso_id references
    from scoring import recompute_all_records
    recompute_all_records()

    return count


# Valid POTA park reference pattern: XX-NNNN or X-NNNN (country code, dash, 4+ digits with leading zeros)
VALID_PARK_PATTERN = re.compile(r'^[A-Z]{1,2}-\d{4,}$')


def detect_park_anomaly(park_ref: str) -> Optional[str]:
    """
    Detect anomalies in a park reference format.

    Valid format: XX-NNNN (e.g., US-0001, K-0001, VE-0123)
    Multiple parks are allowed (comma-separated).

    Returns:
        Anomaly description if found, None if valid format
    """
    if not park_ref:
        return None

    park_ref = park_ref.strip()

    # Handle multiple parks (comma-separated) - check each one
    if ',' in park_ref:
        parks = [p.strip() for p in park_ref.split(',')]
        anomalies = []
        for park in parks:
            anomaly = detect_park_anomaly(park)
            if anomaly:
                anomalies.append(anomaly)
        if anomalies:
            return "; ".join(anomalies)
        return None  # All parks valid

    # Check valid format first
    if VALID_PARK_PATTERN.match(park_ref):
        return None  # Valid format

    # Detect specific anomalies

    # Just a number (missing country prefix entirely)
    if park_ref.isdigit():
        return f"Missing country prefix: '{park_ref}' (expected format: XX-{park_ref.zfill(4)})"

    # Has country code but missing dash (e.g., "US0001")
    match = re.match(r'^([A-Z]{1,2})(\d+)$', park_ref)
    if match:
        country, number = match.groups()
        return f"Missing dash in park reference: '{park_ref}' (expected: {country}-{number.zfill(4)})"

    # Has dash but number not zero-padded properly or too short
    match = re.match(r'^([A-Z]{1,2})-(\d{1,3})$', park_ref)
    if match:
        country, number = match.groups()
        return f"Park number needs leading zeros: '{park_ref}' (expected: {country}-{number.zfill(4)})"

    # Other invalid format
    return f"Invalid park reference format: '{park_ref}' (expected format: XX-NNNN)"


def auto_disqualify_qso(qso_id: int, sport_id: int, reason: str) -> bool:
    """
    Automatically disqualify a QSO for a sport with system-generated reason.

    Args:
        qso_id: The QSO ID to disqualify
        sport_id: The sport ID to disqualify from
        reason: The reason for disqualification

    Returns:
        True if disqualified, False if already disqualified or error
    """
    now = datetime.utcnow().isoformat()

    with get_db() as conn:
        # Check if already disqualified
        cursor = conn.execute(
            "SELECT id, status FROM qso_disqualifications WHERE qso_id = ? AND sport_id = ?",
            (qso_id, sport_id)
        )
        existing = cursor.fetchone()

        if existing:
            if existing["status"] == "disqualified":
                return False  # Already disqualified
            # Update existing record back to disqualified
            conn.execute(
                "UPDATE qso_disqualifications SET status = 'disqualified', updated_at = ? WHERE id = ?",
                (now, existing["id"])
            )
            dq_id = existing["id"]
        else:
            # Create new disqualification
            cursor = conn.execute(
                "INSERT INTO qso_disqualifications (qso_id, sport_id, status, created_at, updated_at) VALUES (?, ?, 'disqualified', ?, ?)",
                (qso_id, sport_id, now, now)
            )
            dq_id = cursor.lastrowid

        # Add comment
        conn.execute(
            "INSERT INTO qso_disqualification_comments (disqualification_id, author_callsign, comment_type, comment, created_at) VALUES (?, ?, 'disqualify', ?, ?)",
            (dq_id, "SYSTEM", reason, now)
        )
        conn.commit()

    return True


def check_and_disqualify_park_anomalies() -> Dict[str, int]:
    """
    Check all QSOs for park reference anomalies and auto-disqualify them.

    Returns:
        Dict with counts of anomalies found and QSOs disqualified
    """
    stats = {
        "anomalies_found": 0,
        "qsos_disqualified": 0,
        "already_disqualified": 0,
    }

    # Get all sports that use park/pota target types (these are the ones affected by park anomalies)
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT id, name FROM sports WHERE target_type IN ('park', 'pota')"
        )
        park_sports = [dict(row) for row in cursor.fetchall()]

        if not park_sports:
            logger.info("No park/pota sports found, skipping anomaly check")
            return stats

        # Find all QSOs with park references that have anomalies
        cursor = conn.execute("""
            SELECT id, competitor_callsign, dx_callsign, my_sig_info, dx_sig_info, qso_datetime_utc
            FROM qsos
            WHERE (my_sig_info IS NOT NULL AND my_sig_info != '')
               OR (dx_sig_info IS NOT NULL AND dx_sig_info != '')
        """)
        qsos_with_parks = cursor.fetchall()

    for qso in qsos_with_parks:
        anomalies = []

        # Check my_sig_info (activator's park)
        if qso["my_sig_info"]:
            anomaly = detect_park_anomaly(qso["my_sig_info"])
            if anomaly:
                anomalies.append(f"MY_SIG_INFO: {anomaly}")

        # Check dx_sig_info (worked station's park)
        if qso["dx_sig_info"]:
            anomaly = detect_park_anomaly(qso["dx_sig_info"])
            if anomaly:
                anomalies.append(f"SIG_INFO: {anomaly}")

        if anomalies:
            stats["anomalies_found"] += 1
            reason = "Auto-flagged: Park reference format anomaly. " + "; ".join(anomalies)

            # Disqualify from all park/pota sports
            for sport in park_sports:
                if auto_disqualify_qso(qso["id"], sport["id"], reason):
                    stats["qsos_disqualified"] += 1
                    logger.info(
                        f"Auto-disqualified QSO {qso['id']} ({qso['competitor_callsign']} -> {qso['dx_callsign']}) "
                        f"from sport '{sport['name']}': {reason}"
                    )
                else:
                    stats["already_disqualified"] += 1

    return stats


async def validate_park_ids(park_ids: Set[str]) -> Set[str]:
    """
    Validate a set of POTA park IDs against the POTA API.

    Returns the set of valid park IDs. Invalid IDs are filtered out.
    Results are cached for 30 days.
    """
    if not park_ids:
        return set()

    valid_ids = set()
    ids_to_check = set()

    # Check cache first
    with get_db() as conn:
        for park_id in park_ids:
            cursor = conn.execute(
                "SELECT reference, cached_at FROM pota_parks WHERE reference = ?",
                (park_id,)
            )
            cached = cursor.fetchone()
            if cached:
                cached_at = datetime.fromisoformat(cached["cached_at"])
                if datetime.utcnow() - cached_at < timedelta(days=30):
                    valid_ids.add(park_id)
                    continue
            ids_to_check.add(park_id)

    # Validate uncached IDs against POTA API
    for park_id in ids_to_check:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://api.pota.app/park/{park_id}",
                    timeout=10.0
                )
                if response.status_code == 200:
                    data = response.json()
                    if data and isinstance(data, dict) and data.get("name"):
                        # Valid park - cache it
                        valid_ids.add(park_id)
                        name = data.get("name", "Unknown")
                        location = data.get("locationDesc", "")
                        grid = data.get("grid", "")
                        with get_db() as conn:
                            conn.execute("""
                                INSERT OR REPLACE INTO pota_parks (reference, name, location, grid, cached_at)
                                VALUES (?, ?, ?, ?, ?)
                            """, (park_id, name, location, grid, datetime.utcnow().isoformat()))
                        logger.info(f"Validated POTA park: {park_id} ({name})")
                    else:
                        logger.warning(f"Invalid POTA park ID: {park_id}")
                else:
                    logger.warning(f"Invalid POTA park ID: {park_id} (HTTP {response.status_code})")
        except httpx.RequestError as e:
            # On network error, tentatively accept the ID
            logger.warning(f"Could not validate park {park_id}: {e}")
            valid_ids.add(park_id)

    return valid_ids


def _collect_park_ids(qsos: List[QSOData]) -> Set[str]:
    """Collect all unique park IDs from a list of QSOs."""
    park_ids = set()
    for qso in qsos:
        if qso.my_sig_info:
            park_ids.add(qso.my_sig_info)
        if qso.dx_sig_info:
            park_ids.add(qso.dx_sig_info)
    return park_ids


def _filter_invalid_park_ids(qsos: List[QSOData], valid_ids: Set[str]) -> List[QSOData]:
    """
    Return a new list of QSOs with invalid park IDs cleared.

    Since QSOData is a dataclass, we need to create new instances.
    """
    from dataclasses import replace
    filtered = []
    for qso in qsos:
        my_sig = qso.my_sig_info if qso.my_sig_info in valid_ids else None
        dx_sig = qso.dx_sig_info if qso.dx_sig_info in valid_ids else None
        if my_sig != qso.my_sig_info or dx_sig != qso.dx_sig_info:
            # Need to create a new QSOData with filtered values
            qso = replace(qso, my_sig_info=my_sig, dx_sig_info=dx_sig)
        filtered.append(qso)
    return filtered


async def populate_competitor_name(callsign: str) -> bool:
    """
    Look up and populate a competitor's name from callsign lookup if not already set.

    Args:
        callsign: Competitor callsign

    Returns:
        True if name was updated, False otherwise
    """
    from callsign_lookup import lookup_callsign

    with get_db() as conn:
        # Check if name is already set
        cursor = conn.execute(
            "SELECT first_name, last_name FROM competitors WHERE callsign = ?",
            (callsign.upper(),)
        )
        row = cursor.fetchone()
        if not row:
            return False

        # If name already set, skip lookup
        if row["first_name"]:
            return False

        # Look up name
        info = await lookup_callsign(callsign)
        if info and info.first_name:
            conn.execute(
                "UPDATE competitors SET first_name = ?, last_name = ? WHERE callsign = ?",
                (info.first_name, info.last_name, callsign.upper())
            )
            return True

    return False


async def sync_competitor(callsign: str, recompute: bool = True) -> dict:
    """
    Sync a single competitor's QSOs from QRZ using stored API key.

    Args:
        callsign: Competitor callsign
        recompute: If True, recompute medals and send notifications after sync.
                   Set to False during bulk sync to defer recomputation.

    Returns:
        dict with sync results
    """
    with get_db() as conn:
        # Get competitor
        cursor = conn.execute(
            "SELECT * FROM competitors WHERE callsign = ?",
            (callsign.upper(),)
        )
        competitor = cursor.fetchone()

        if not competitor:
            return {"error": f"Competitor {callsign} not found"}

        # Check if API key is configured
        if not competitor["qrz_api_key_encrypted"]:
            return {"error": "QRZ API key not configured. Please add your API key in Settings."}

        # Decrypt API key
        try:
            api_key = decrypt_api_key(competitor["qrz_api_key_encrypted"])
        except (InvalidToken, binascii.Error, ValueError) as e:
            logger.error(f"Failed to decrypt API key for {callsign}: {e}")
            return {"error": "Failed to decrypt API key. Please re-enter your QRZ API key in Settings."}

    # Use the shared sync function with the decrypted key
    return await sync_competitor_with_key(callsign, api_key, recompute=recompute)


async def sync_competitor_with_key(callsign: str, api_key: str, recompute: bool = True) -> dict:
    """
    Sync a single competitor's QSOs from QRZ using provided API key.

    Uses incremental sync to reduce API calls:
    - First sync: fetches QSOs from olympiad start date
    - Subsequent syncs: fetches from 60 days before last sync to catch confirmations

    Args:
        callsign: Competitor callsign
        api_key: QRZ API key (not encrypted)
        recompute: If True, recompute medals and send notifications after sync.
                   Set to False during bulk sync to defer recomputation.

    Returns:
        dict with sync results
    """
    # Try to populate name if not set
    try:
        await populate_competitor_name(callsign)
    except Exception as e:
        logger.warning(f"Failed to populate name for {callsign}: {e}")

    # Determine sync strategy
    # The QRZ BETWEEN date filter is unreliable for some accounts (returns COUNT=0).
    # We use AFTERLOGID for incremental syncs and do a full re-fetch periodically
    # to catch confirmation updates on existing QSOs.
    max_logid = 0
    has_unconfirmed = False
    with get_db() as conn:
        # Get highest logid and check for unconfirmed QSOs
        cursor = conn.execute(
            """SELECT
                MAX(CAST(qrz_logid AS INTEGER)) as max_logid,
                COUNT(*) as total,
                SUM(CASE WHEN is_confirmed = 0 THEN 1 ELSE 0 END) as unconfirmed
               FROM qsos WHERE competitor_callsign = ?""",
            (callsign.upper(),)
        )
        row = cursor.fetchone()
        if row:
            max_logid = row["max_logid"] or 0
            has_unconfirmed = (row["unconfirmed"] or 0) > 0

    # If user has unconfirmed QSOs, do a full fetch to check for confirmation updates
    # Otherwise, just fetch new QSOs after the last known logid
    if max_logid == 0:
        logger.info(f"First sync for {callsign}: fetching all QSOs")
        after_logid = 0
    elif has_unconfirmed:
        # Full re-fetch to catch confirmation updates
        logger.info(f"Sync for {callsign}: full fetch to check confirmations (has unconfirmed QSOs)")
        after_logid = 0
    else:
        # All QSOs are confirmed, just get new ones
        logger.info(f"Incremental sync for {callsign}: fetching QSOs after logid {max_logid}")
        after_logid = max_logid

    # Fetch QSOs from QRZ using AFTERLOGID (more reliable than BETWEEN date filter)
    try:
        qsos = await fetch_qsos(api_key, confirmed_only=False, after_logid=after_logid)
    except QRZAPIError as e:
        return {"error": str(e)}

    if not qsos:
        # Still update last_sync_at even when no QSOs found - the sync did complete successfully
        with get_db() as conn:
            conn.execute(
                "UPDATE competitors SET last_sync_at = ? WHERE callsign = ?",
                (datetime.utcnow().isoformat(), callsign.upper())
            )
            conn.commit()
        return {"message": "No new QSOs found in QRZ logbook", "new_qsos": 0, "updated_qsos": 0}

    # Validate park IDs against POTA API
    park_ids = _collect_park_ids(qsos)
    if park_ids:
        valid_park_ids = await validate_park_ids(park_ids)
        qsos = _filter_invalid_park_ids(qsos, valid_park_ids)

    # Process and store QSOs in small batches to avoid blocking reads
    new_count = 0
    updated_count = 0
    newly_confirmed_calls = []  # Track DX callsigns for newly confirmed QSOs
    batch_size = 50

    for i in range(0, len(qsos), batch_size):
        batch = qsos[i:i + batch_size]
        with get_db() as conn:
            for qso in batch:
                result = _upsert_qso(conn, callsign.upper(), qso)
                if result["status"] == "new":
                    new_count += 1
                elif result["status"] == "updated":
                    updated_count += 1
                # Track newly confirmed QSOs for notification
                if result.get("newly_confirmed"):
                    newly_confirmed_calls.append(result["dx_callsign"])
            conn.commit()

    # Update last sync time
    with get_db() as conn:
        conn.execute(
            "UPDATE competitors SET last_sync_at = ? WHERE callsign = ?",
            (datetime.utcnow().isoformat(), callsign.upper())
        )
        conn.commit()

    if recompute:
        # Get medal state BEFORE recomputation for comparison
        old_medals = _get_competitor_medal_state(callsign.upper())

        # Recompute medals for all active matches after sync
        recompute_all_active_matches()

        # Get medal state AFTER recomputation
        new_medals = _get_competitor_medal_state(callsign.upper())

        # Send notifications for new confirmations
        if newly_confirmed_calls:
            _notify_new_confirmations(callsign.upper(), newly_confirmed_calls)

        # Look up first name for Discord display
        with get_db() as conn:
            row = conn.execute("SELECT first_name FROM competitors WHERE callsign = ?",
                               (callsign.upper(),)).fetchone()
            first_name = row["first_name"] if row else None

        # Send push notifications for medal changes (Discord deferred for batching)
        medal_changes = _notify_medal_changes(callsign.upper(), old_medals, new_medals,
                                              discord=False, first_name=first_name)

        # Send ONE batched Discord message for all medal changes
        if medal_changes:
            try:
                from notifications import discord_notify_sync_summary
                discord_notify_sync_summary(medal_changes, [])
            except Exception as e:
                logger.error(f"Failed to send Discord medal summary: {e}")

        # Send email notifications for any new medals
        try:
            from email_service import notify_new_medals
            await notify_new_medals()
        except Exception as e:
            logger.error(f"Failed to send medal notification emails: {e}")

    return {
        "callsign": callsign.upper(),
        "new_qsos": new_count,
        "updated_qsos": updated_count,
        "total_fetched": len(qsos),
        "new_confirmations": len(newly_confirmed_calls),
        "newly_confirmed_calls": newly_confirmed_calls,
    }


async def sync_competitor_lotw_stored(callsign: str) -> dict:
    """
    Sync a competitor's QSOs from LoTW using stored credentials.

    Args:
        callsign: Competitor callsign

    Returns:
        dict with sync results
    """
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT lotw_username_encrypted, lotw_password_encrypted FROM competitors WHERE callsign = ?",
            (callsign.upper(),)
        )
        competitor = cursor.fetchone()

        if not competitor:
            return {"error": f"Competitor {callsign} not found"}

        if not competitor["lotw_username_encrypted"] or not competitor["lotw_password_encrypted"]:
            return {"error": "LoTW credentials not configured. Please add your credentials in Settings."}

        try:
            lotw_username = decrypt_api_key(competitor["lotw_username_encrypted"])
            lotw_password = decrypt_api_key(competitor["lotw_password_encrypted"])
        except (InvalidToken, binascii.Error, ValueError) as e:
            logger.error(f"Failed to decrypt LoTW credentials for {callsign}: {e}")
            return {"error": "Failed to decrypt LoTW credentials. Please re-enter your credentials in Settings."}

    return await sync_competitor_lotw(callsign, lotw_username, lotw_password)


async def sync_competitor_lotw(callsign: str, lotw_username: str, lotw_password: str) -> dict:
    """
    Sync a competitor's QSOs from LoTW using provided credentials.

    Uses incremental sync to reduce API calls:
    - First sync: fetches QSOs from olympiad start date
    - Subsequent syncs: fetches from 60 days before last sync to catch confirmations

    Args:
        callsign: Competitor callsign
        lotw_username: LoTW username
        lotw_password: LoTW password

    Returns:
        dict with sync results
    """
    # Try to populate name if not set
    try:
        await populate_competitor_name(callsign)
    except Exception as e:
        logger.warning(f"Failed to populate name for {callsign}: {e}")

    # Calculate date range for incremental sync
    start_date_str = None
    with get_db() as conn:
        # Verify competitor exists
        cursor = conn.execute(
            "SELECT callsign FROM competitors WHERE callsign = ?",
            (callsign.upper(),)
        )
        competitor = cursor.fetchone()
        if not competitor:
            return {"error": f"Competitor {callsign} not found"}

        # Get most recent QSO datetime for this competitor
        cursor = conn.execute(
            "SELECT MAX(qso_datetime_utc) FROM qsos WHERE competitor_callsign = ?",
            (callsign.upper(),)
        )
        most_recent_qso = cursor.fetchone()[0]

        # Get active olympiad start date
        cursor = conn.execute(
            "SELECT start_date FROM olympiads WHERE is_active = 1"
        )
        olympiad = cursor.fetchone()

        if olympiad:
            olympiad_start = datetime.fromisoformat(olympiad["start_date"])
        else:
            # No active olympiad, use a reasonable default (1 year ago)
            olympiad_start = datetime.utcnow() - timedelta(days=365)

        if most_recent_qso:
            # Incremental sync: go back 60 days from most recent QSO to catch confirmations
            # and any QSOs that were added to LoTW after the fact
            latest_qso = datetime.fromisoformat(most_recent_qso)
            confirmation_window = latest_qso - timedelta(days=60)
            # Use the later of olympiad start or confirmation window
            since_date = max(olympiad_start, confirmation_window)
            start_date_str = since_date.strftime("%Y-%m-%d")
            logger.info(f"Incremental LoTW sync for {callsign}: fetching QSOs from {since_date.date()} (most recent QSO: {latest_qso.date()})")
        else:
            # First sync: pull ALL QSOs to seed the database
            start_date_str = None
            logger.info(f"First LoTW sync for {callsign}: fetching all QSOs (no date filter)")

    # Fetch QSOs from LoTW with date range
    try:
        qsos = await fetch_lotw_qsos(lotw_username, lotw_password, confirmed_only=False, start_date=start_date_str)
    except LoTWError as e:
        return {"error": str(e)}

    if not qsos:
        return {"message": "No new QSOs found in LoTW", "new_qsos": 0, "updated_qsos": 0}

    # Validate park IDs against POTA API
    park_ids = _collect_park_ids(qsos)
    if park_ids:
        valid_park_ids = await validate_park_ids(park_ids)
        qsos = _filter_invalid_park_ids(qsos, valid_park_ids)

    # Process and store QSOs in small batches to avoid blocking reads
    new_count = 0
    updated_count = 0
    newly_confirmed_calls = []  # Track DX callsigns for newly confirmed QSOs
    batch_size = 50

    for i in range(0, len(qsos), batch_size):
        batch = qsos[i:i + batch_size]
        with get_db() as conn:
            for qso in batch:
                result = _upsert_qso(conn, callsign.upper(), qso)
                if result["status"] == "new":
                    new_count += 1
                elif result["status"] == "updated":
                    updated_count += 1
                # Track newly confirmed QSOs for notification
                if result.get("newly_confirmed"):
                    newly_confirmed_calls.append(result["dx_callsign"])
            conn.commit()

    # Update last sync time
    with get_db() as conn:
        conn.execute(
            "UPDATE competitors SET last_sync_at = ? WHERE callsign = ?",
            (datetime.utcnow().isoformat(), callsign.upper())
        )
        conn.commit()

    # Get medal state BEFORE recomputation for comparison
    old_medals = _get_competitor_medal_state(callsign.upper())

    # Recompute medals for all active matches after sync
    recompute_all_active_matches()

    # Get medal state AFTER recomputation
    new_medals = _get_competitor_medal_state(callsign.upper())

    # Send notifications for new confirmations
    if newly_confirmed_calls:
        _notify_new_confirmations(callsign.upper(), newly_confirmed_calls)

    # Look up first name for Discord display
    with get_db() as conn:
        row = conn.execute("SELECT first_name FROM competitors WHERE callsign = ?",
                           (callsign.upper(),)).fetchone()
        first_name = row["first_name"] if row else None

    # Send push notifications for medal changes (Discord deferred for batching)
    medal_changes = _notify_medal_changes(callsign.upper(), old_medals, new_medals,
                                          discord=False, first_name=first_name)

    # Send ONE batched Discord message for all medal changes
    if medal_changes:
        try:
            from notifications import discord_notify_sync_summary
            discord_notify_sync_summary(medal_changes, [])
        except Exception as e:
            logger.error(f"Failed to send Discord medal summary: {e}")

    # Send email notifications for any new medals
    try:
        from email_service import notify_new_medals
        await notify_new_medals()
    except Exception as e:
        logger.error(f"Failed to send medal notification emails: {e}")

    return {
        "callsign": callsign.upper(),
        "new_qsos": new_count,
        "updated_qsos": updated_count,
        "total_fetched": len(qsos),
        "new_confirmations": len(newly_confirmed_calls),
    }


def _upsert_qso(conn, competitor_callsign: str, qso: QSOData) -> dict:
    """
    Insert or update a QSO in the database.

    Returns:
        dict with keys:
        - status: "new", "updated", or None
        - newly_confirmed: True if QSO became confirmed for first time
        - dx_callsign: DX station callsign (for notifications)
    """
    # Check if QSO already exists (by QRZ logid or by unique combination)
    existing = None
    if qso.qrz_logid:
        cursor = conn.execute(
            "SELECT id, is_confirmed FROM qsos WHERE qrz_logid = ? AND competitor_callsign = ?",
            (qso.qrz_logid, competitor_callsign)
        )
        existing = cursor.fetchone()

    if existing is None:
        # Check by callsign + datetime + mode (with 2-minute tolerance for different sources)
        # Mode must match to avoid merging FT4/FT8 or SSB/CW contacts
        # For sig_info fields (two-fer support): only consider QSOs different if BOTH have
        # non-NULL but different sig_info values. NULL matches anything (same QSO, missing data).
        qso_dt = qso.qso_datetime.isoformat()
        cursor = conn.execute(
            """SELECT id, is_confirmed FROM qsos
               WHERE competitor_callsign = ? AND dx_callsign = ?
               AND (mode = ? OR mode IS NULL OR ? IS NULL)
               AND (my_sig_info IS NULL OR ? IS NULL OR my_sig_info = ?)
               AND (dx_sig_info IS NULL OR ? IS NULL OR dx_sig_info = ?)
               AND ABS(CAST((julianday(qso_datetime_utc) - julianday(?)) * 86400 AS INTEGER)) <= 120""",
            (competitor_callsign, qso.dx_callsign, qso.mode, qso.mode,
             qso.my_sig_info, qso.my_sig_info, qso.dx_sig_info, qso.dx_sig_info, qso_dt)
        )
        existing = cursor.fetchone()

    # Calculate distance and cool factor
    distance_km = None
    cool_factor = None

    my_grid = qso.my_grid
    dx_grid = qso.dx_grid

    if my_grid and dx_grid:
        try:
            distance_km = grid_distance(my_grid, dx_grid)
            if qso.tx_power and qso.tx_power > 0:
                cool_factor = distance_km / qso.tx_power
        except ValueError:
            pass  # Invalid grid format

    if existing:
        # Check if this update will result in a new confirmation
        was_confirmed = bool(existing["is_confirmed"])
        will_be_confirmed = qso.is_confirmed
        newly_confirmed = not was_confirmed and will_be_confirmed

        # Merge fields - use COALESCE to keep existing values if new value is NULL
        # This allows QRZ (with tx_power) and LoTW (with confirmation) to complement each other
        # Set confirmed_at timestamp when QSO becomes confirmed for the first time
        conn.execute("""
            UPDATE qsos SET
                is_confirmed = CASE WHEN ? = 1 THEN 1 ELSE is_confirmed END,
                confirmed_at = CASE WHEN ? = 1 AND is_confirmed = 0 AND confirmed_at IS NULL THEN ? ELSE confirmed_at END,
                band = COALESCE(?, band),
                mode = COALESCE(?, mode),
                tx_power_w = COALESCE(?, tx_power_w),
                my_dxcc = COALESCE(?, my_dxcc),
                my_grid = COALESCE(?, my_grid),
                my_sig_info = COALESCE(?, my_sig_info),
                dx_dxcc = COALESCE(?, dx_dxcc),
                dx_grid = COALESCE(?, dx_grid),
                dx_sig_info = COALESCE(?, dx_sig_info),
                distance_km = COALESCE(?, distance_km),
                cool_factor = COALESCE(?, cool_factor)
            WHERE id = ?
        """, (
            1 if qso.is_confirmed else 0,
            1 if qso.is_confirmed else 0,
            datetime.utcnow().isoformat(),
            qso.band,
            qso.mode,
            qso.tx_power,
            qso.my_dxcc,
            qso.my_grid,
            qso.my_sig_info,
            qso.dx_dxcc,
            qso.dx_grid,
            qso.dx_sig_info,
            distance_km,
            cool_factor,
            existing["id"],
        ))
        return {"status": "updated", "newly_confirmed": newly_confirmed, "dx_callsign": qso.dx_callsign}
    else:
        # Insert new QSO
        import sqlite3
        try:
            cursor = conn.execute("""
                INSERT INTO qsos (
                    competitor_callsign, dx_callsign, qso_datetime_utc,
                    band, mode, tx_power_w,
                    my_dxcc, my_grid, my_sig_info,
                    dx_dxcc, dx_grid, dx_sig_info,
                    distance_km, cool_factor, is_confirmed, confirmed_at, qrz_logid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                competitor_callsign,
                qso.dx_callsign,
                qso.qso_datetime.isoformat(),
                qso.band,
                qso.mode,
                qso.tx_power,
                qso.my_dxcc,
                qso.my_grid,
                qso.my_sig_info,
                qso.dx_dxcc,
                qso.dx_grid,
                qso.dx_sig_info,
                distance_km,
                cool_factor,
                1 if qso.is_confirmed else 0,
                datetime.utcnow().isoformat() if qso.is_confirmed else None,
                qso.qrz_logid,
            ))
        except sqlite3.IntegrityError:
            # Duplicate QSO - race condition. Find the existing record and merge.
            # Both sig_info fields are part of uniqueness (supports two-fers where same contact counts for multiple parks)
            logger.info(f"Duplicate detected, merging: {competitor_callsign} -> {qso.dx_callsign} at {qso.qso_datetime} (my_park: {qso.my_sig_info}, dx_park: {qso.dx_sig_info})")
            cursor = conn.execute(
                """SELECT id FROM qsos
                   WHERE competitor_callsign = ? AND dx_callsign = ?
                   AND (my_sig_info IS NULL OR ? IS NULL OR my_sig_info = ?)
                   AND (dx_sig_info IS NULL OR ? IS NULL OR dx_sig_info = ?)
                   AND ABS(CAST((julianday(qso_datetime_utc) - julianday(?)) * 86400 AS INTEGER)) <= 120""",
                (competitor_callsign, qso.dx_callsign, qso.my_sig_info,
                 qso.my_sig_info, qso.dx_sig_info, qso.dx_sig_info, qso.qso_datetime.isoformat())
            )
            existing_row = cursor.fetchone()
            if existing_row:
                # Merge into existing record
                conn.execute("""
                    UPDATE qsos SET
                        is_confirmed = CASE WHEN ? = 1 THEN 1 ELSE is_confirmed END,
                        confirmed_at = CASE WHEN ? = 1 AND is_confirmed = 0 AND confirmed_at IS NULL THEN ? ELSE confirmed_at END,
                        band = COALESCE(?, band),
                        mode = COALESCE(?, mode),
                        tx_power_w = COALESCE(?, tx_power_w),
                        my_dxcc = COALESCE(?, my_dxcc),
                        my_grid = COALESCE(?, my_grid),
                        my_sig_info = COALESCE(?, my_sig_info),
                        dx_dxcc = COALESCE(?, dx_dxcc),
                        dx_grid = COALESCE(?, dx_grid),
                        dx_sig_info = COALESCE(?, dx_sig_info),
                        distance_km = COALESCE(?, distance_km),
                        cool_factor = COALESCE(?, cool_factor),
                        qrz_logid = COALESCE(?, qrz_logid)
                    WHERE id = ?
                """, (
                    1 if qso.is_confirmed else 0,
                    1 if qso.is_confirmed else 0,
                    datetime.utcnow().isoformat(),
                    qso.band,
                    qso.mode,
                    qso.tx_power,
                    qso.my_dxcc,
                    qso.my_grid,
                    qso.my_sig_info,
                    qso.dx_dxcc,
                    qso.dx_grid,
                    qso.dx_sig_info,
                    distance_km,
                    cool_factor,
                    qso.qrz_logid,
                    existing_row["id"],
                ))
                return {"status": "updated", "newly_confirmed": False, "dx_callsign": qso.dx_callsign}
            return {"status": None, "newly_confirmed": False, "dx_callsign": qso.dx_callsign}

        # Note: Records are updated via recompute_all_records() which only
        # considers QSOs that qualified for matches (correct target, time period, mode)

        # New QSO that's already confirmed counts as newly confirmed
        return {"status": "new", "newly_confirmed": qso.is_confirmed, "dx_callsign": qso.dx_callsign}


def merge_duplicate_qsos() -> dict:
    """
    Find and merge duplicate QSOs in the database.

    Duplicates are identified by: competitor_callsign + dx_callsign + similar timestamp (within 2 minutes)
    When merging, we keep the best value for each field (non-NULL preferred).

    Returns:
        dict with merge statistics
    """
    from database import get_db
    from datetime import datetime

    merged_count = 0
    deleted_count = 0

    with get_db() as conn:
        # Get all QSOs grouped by competitor, dx_callsign, and mode, ordered by time
        # Mode must match to avoid merging FT4/FT8 or SSB/CW contacts
        cursor = conn.execute("""
            SELECT id, competitor_callsign, dx_callsign, mode, qso_datetime_utc, my_sig_info, dx_sig_info
            FROM qsos
            ORDER BY competitor_callsign, dx_callsign, mode, qso_datetime_utc
        """)
        all_qsos = [dict(row) for row in cursor.fetchall()]

        # Find duplicates within 2-minute window (same callsigns, mode, compatible sig_info fields)
        # For sig_info: NULL matches anything (same QSO, just missing data from one source).
        # Only treat as different QSOs (two-fers) when BOTH have non-NULL but different values.
        duplicates_to_merge = []
        i = 0
        while i < len(all_qsos):
            qso = all_qsos[i]
            group = [qso["id"]]

            # Look ahead for QSOs with same callsigns, mode, compatible sig_info within 2 minutes
            j = i + 1
            while j < len(all_qsos):
                next_qso = all_qsos[j]
                # Must match on callsigns and mode
                if (next_qso["competitor_callsign"] != qso["competitor_callsign"] or
                    next_qso["dx_callsign"] != qso["dx_callsign"] or
                    next_qso["mode"] != qso["mode"]):
                    break

                # For sig_info: NULL matches anything; only different when both non-NULL and different
                my_sig_a = qso["my_sig_info"]
                my_sig_b = next_qso["my_sig_info"]
                dx_sig_a = qso["dx_sig_info"]
                dx_sig_b = next_qso["dx_sig_info"]
                if (my_sig_a and my_sig_b and my_sig_a != my_sig_b):
                    break
                if (dx_sig_a and dx_sig_b and dx_sig_a != dx_sig_b):
                    break

                # Check time difference
                dt1 = datetime.fromisoformat(qso["qso_datetime_utc"])
                dt2 = datetime.fromisoformat(next_qso["qso_datetime_utc"])
                if abs((dt2 - dt1).total_seconds()) <= 120:  # 2 minutes
                    group.append(next_qso["id"])
                    j += 1
                else:
                    break

            if len(group) > 1:
                duplicates_to_merge.append(group)
                i = j  # Skip past the group
            else:
                i += 1

        # Now merge each duplicate group
        for id_group in duplicates_to_merge:
            # Fetch full QSO data for this group
            placeholders = ",".join("?" * len(id_group))
            cursor = conn.execute(f"SELECT * FROM qsos WHERE id IN ({placeholders}) ORDER BY id", id_group)
            qsos = [dict(row) for row in cursor.fetchall()]

            if len(qsos) < 2:
                continue

            # Keep the first one as the master, merge others into it
            master_id = qsos[0]["id"]

            for other in qsos[1:]:
                # Delete the duplicate FIRST to avoid unique index collision
                # (the update may set sig_info values that match the duplicate's unique key)
                conn.execute("DELETE FROM qsos WHERE id = ?", (other["id"],))
                deleted_count += 1

                # Merge each field - take non-NULL value, prefer confirmed
                # For confirmed_at, take the earliest timestamp if both are confirmed
                conn.execute("""
                    UPDATE qsos SET
                        is_confirmed = CASE WHEN ? = 1 OR is_confirmed = 1 THEN 1 ELSE 0 END,
                        confirmed_at = CASE
                            WHEN confirmed_at IS NULL THEN ?
                            WHEN ? IS NULL THEN confirmed_at
                            WHEN ? < confirmed_at THEN ?
                            ELSE confirmed_at
                        END,
                        band = COALESCE(band, ?),
                        mode = COALESCE(mode, ?),
                        tx_power_w = COALESCE(tx_power_w, ?),
                        my_dxcc = COALESCE(my_dxcc, ?),
                        my_grid = COALESCE(my_grid, ?),
                        my_sig_info = COALESCE(my_sig_info, ?),
                        dx_dxcc = COALESCE(dx_dxcc, ?),
                        dx_grid = COALESCE(dx_grid, ?),
                        dx_sig_info = COALESCE(dx_sig_info, ?),
                        distance_km = COALESCE(distance_km, ?),
                        cool_factor = COALESCE(cool_factor, ?),
                        qrz_logid = COALESCE(qrz_logid, ?)
                    WHERE id = ?
                """, (
                    other["is_confirmed"],
                    other.get("confirmed_at"),
                    other.get("confirmed_at"),
                    other.get("confirmed_at"),
                    other.get("confirmed_at"),
                    other["band"],
                    other["mode"],
                    other["tx_power_w"],
                    other["my_dxcc"],
                    other["my_grid"],
                    other["my_sig_info"],
                    other["dx_dxcc"],
                    other["dx_grid"],
                    other["dx_sig_info"],
                    other["distance_km"],
                    other["cool_factor"],
                    other["qrz_logid"],
                    master_id,
                ))

            merged_count += 1

    # Recompute records from scratch to fix any orphaned qso_id references
    if deleted_count > 0:
        from scoring import recompute_all_records
        recompute_all_records()

    return {
        "duplicate_groups": merged_count,
        "qsos_deleted": deleted_count,
    }


async def sync_all_competitors() -> dict:
    """
    Sync all competitors and recompute medals.

    Uses batch mode: syncs all competitors without per-competitor recomputation,
    then does ONE recomputation and ONE batch of notifications at the end.

    Returns:
        dict with sync summary
    """
    with get_db() as conn:
        # Get all competitors with first names for display
        cursor = conn.execute("SELECT callsign, first_name FROM competitors")
        competitors = {row["callsign"]: row["first_name"] for row in cursor.fetchall()}
    callsigns = list(competitors.keys())

    # Capture medal state BEFORE sync for ALL competitors
    old_medal_states = {}
    for callsign in callsigns:
        old_medal_states[callsign] = _get_competitor_medal_state(callsign)

    # Sync all competitors WITHOUT per-competitor recomputation/notifications
    results = []
    for callsign in callsigns:
        result = await sync_competitor(callsign, recompute=False)
        results.append(result)

    # Populate missing dx_dxcc values from callsign lookups
    try:
        dxcc_result = await populate_missing_dxcc()
        logger.info(f"Populated missing dx_dxcc: {dxcc_result}")
    except Exception as e:
        logger.warning(f"Failed to populate missing dx_dxcc: {e}")

    # ONE recomputation for all active matches (run in thread pool to avoid blocking event loop)
    import asyncio
    await asyncio.to_thread(recompute_all_active_matches)

    # Capture medal state AFTER recomputation and send ONE batch of notifications
    all_medal_changes = []
    for i, callsign in enumerate(callsigns):
        result = results[i]
        if "error" in result:
            continue

        new_medals = _get_competitor_medal_state(callsign)
        old_medals = old_medal_states[callsign]

        # Send notifications for new confirmations
        newly_confirmed_calls = result.get("newly_confirmed_calls", [])
        if newly_confirmed_calls:
            _notify_new_confirmations(callsign, newly_confirmed_calls)

        # Collect medal changes (push notifications sent, Discord deferred)
        changes = _notify_medal_changes(callsign, old_medals, new_medals,
                                        discord=False, first_name=competitors.get(callsign))
        all_medal_changes.extend(changes)

    # Send ONE consolidated Discord message for all medal changes
    if all_medal_changes:
        try:
            from notifications import discord_notify_sync_summary
            discord_notify_sync_summary(all_medal_changes, [])
        except Exception as e:
            logger.error(f"Failed to send Discord sync summary: {e}")

    # Send email notifications for any new medals
    try:
        from email_service import notify_new_medals
        email_result = await notify_new_medals()
        if email_result["sent"] > 0:
            logger.info(f"Sent {email_result['sent']} medal notification emails ({email_result['medals_notified']} medals)")
    except Exception as e:
        logger.error(f"Failed to send medal notification emails: {e}")

    total_new = sum(r.get("new_qsos", 0) for r in results)
    total_updated = sum(r.get("updated_qsos", 0) for r in results)
    errors = [r for r in results if "error" in r]

    return {
        "competitors_synced": len(callsigns),
        "total_new_qsos": total_new,
        "total_updated_qsos": total_updated,
        "errors": errors,
    }


def recompute_all_active_matches():
    """Recompute medals for all matches in the active Olympiad."""
    # First, check for and auto-disqualify QSOs with park reference anomalies
    anomaly_stats = check_and_disqualify_park_anomalies()
    if anomaly_stats["anomalies_found"] > 0:
        logger.info(
            f"Park anomaly check: {anomaly_stats['anomalies_found']} anomalies found, "
            f"{anomaly_stats['qsos_disqualified']} QSOs disqualified, "
            f"{anomaly_stats['already_disqualified']} already disqualified"
        )

    with get_db() as conn:
        # Get all matches and sports in active Olympiad
        cursor = conn.execute("""
            SELECT m.id as match_id, s.id as sport_id FROM matches m
            JOIN sports s ON m.sport_id = s.id
            JOIN olympiads o ON s.olympiad_id = o.id
            WHERE o.is_active = 1
        """)
        match_sport_pairs = [(row["match_id"], row["sport_id"]) for row in cursor.fetchall()]

        # Get unique sport IDs
        sport_ids = list(set(sp[1] for sp in match_sport_pairs))

    # Recompute individual medals (suppress notifications - no new data arrived)
    for match_id, _ in match_sport_pairs:
        recompute_match_medals(match_id, notify=False)

    # Recompute team standings for each sport
    for sport_id in sport_ids:
        compute_team_standings(sport_id)
        # Sport-level done, now do match-level for matches in this sport
        sport_matches = [m for m, s in match_sport_pairs if s == sport_id]
        for match_id in sport_matches:
            compute_team_standings(sport_id, match_id)

    # Pre-compute records cache for the /records page
    precompute_records_cache()

    # Regenerate cached PDF once after all medals/records are updated
    from pdf_export import regenerate_active_olympiad_pdf
    regenerate_active_olympiad_pdf()


def recompute_sport_matches(sport_id: int):
    """Recompute medals for all matches in a specific sport."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT id FROM matches WHERE sport_id = ?",
            (sport_id,)
        )
        match_ids = [row["id"] for row in cursor.fetchall()]

    for match_id in match_ids:
        recompute_match_medals(match_id, notify=False)

    # Recompute team standings for this sport
    compute_team_standings(sport_id)
    for match_id in match_ids:
        compute_team_standings(sport_id, match_id)

    # Pre-compute records cache for the /records page
    precompute_records_cache()

    # Regenerate cached PDF once after all medals/records are updated
    from pdf_export import regenerate_active_olympiad_pdf
    regenerate_active_olympiad_pdf()


async def populate_missing_dxcc() -> dict:
    """
    Populate missing dx_dxcc values by looking up callsigns.

    Uses the callsign lookup service (QRZ/HamQTH) to get DXCC codes
    for QSOs that are missing dx_dxcc.

    Returns:
        dict with update statistics
    """
    from callsign_lookup import lookup_callsign

    with get_db() as conn:
        # Get all unique dx_callsigns that are missing dx_dxcc
        cursor = conn.execute("""
            SELECT DISTINCT dx_callsign FROM qsos
            WHERE dx_dxcc IS NULL AND dx_callsign IS NOT NULL
            LIMIT 100
        """)
        missing_callsigns = [row["dx_callsign"] for row in cursor.fetchall()]

    if not missing_callsigns:
        return {"updated": 0, "lookups": 0}

    lookups = 0
    updated = 0

    for callsign in missing_callsigns:
        try:
            lookups += 1
            info = await lookup_callsign(callsign)

            if info and info.dxcc:
                with get_db() as conn:
                    # Update all QSOs with this dx_callsign that are missing dx_dxcc
                    cursor = conn.execute(
                        "UPDATE qsos SET dx_dxcc = ? WHERE dx_callsign = ? AND dx_dxcc IS NULL",
                        (info.dxcc, callsign)
                    )
                    updated += cursor.rowcount
                    logger.info(f"Updated dx_dxcc for {callsign}: DXCC={info.dxcc} ({cursor.rowcount} QSOs)")
        except Exception as e:
            logger.warning(f"Failed to look up {callsign}: {e}")

    return {"updated": updated, "lookups": lookups, "callsigns": len(missing_callsigns)}


# ============================================================
# PUSH NOTIFICATION HELPERS
# ============================================================

def _get_competitor_medal_state(callsign: str) -> dict:
    """
    Get the current medal state for a competitor across all active matches.

    Returns dict keyed by match_id with medal info.
    """
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT m.match_id, m.role, m.qso_race_medal, m.cool_factor_medal, m.total_points,
                   mt.target_value, s.name as sport_name
            FROM medals m
            JOIN matches mt ON m.match_id = mt.id
            JOIN sports s ON mt.sport_id = s.id
            JOIN olympiads o ON s.olympiad_id = o.id
            WHERE m.callsign = ? AND o.is_active = 1
        """, (callsign,))

        result = {}
        for row in cursor.fetchall():
            key = f"{row['match_id']}-{row['role']}"
            result[key] = {
                "match_id": row["match_id"],
                "role": row["role"],
                "qso_race_medal": row["qso_race_medal"],
                "cool_factor_medal": row["cool_factor_medal"],
                "total_points": row["total_points"],
                "target_value": row["target_value"],
                "sport_name": row["sport_name"],
            }
        return result


def _notify_new_confirmations(callsign: str, dx_callsigns: list):
    """Send push notification for new QSO confirmations."""
    try:
        from notifications import notify_new_confirmations
        notify_new_confirmations(callsign, len(dx_callsigns), dx_callsigns[:5])
    except Exception as e:
        logger.error(f"Failed to send confirmation notification for {callsign}: {e}")


def _notify_medal_changes(callsign: str, old_medals: dict, new_medals: dict,
                          discord: bool = True, first_name: str = None) -> list:
    """Send push notifications for medal changes.

    Args:
        callsign: The competitor's callsign
        old_medals: Medal state before sync
        new_medals: Medal state after sync
        discord: If True, send individual Discord notifications. If False, skip
                 Discord (for batch mode where a summary is sent instead).
        first_name: Optional first name for Discord display

    Returns:
        List of medal change dicts for batch summary use.
    """
    changes = []
    try:
        from notifications import notify_medal_change, discord_notify_medal

        # Find matches where medals changed
        all_keys = set(old_medals.keys()) | set(new_medals.keys())

        for key in all_keys:
            old = old_medals.get(key, {})
            new = new_medals.get(key, {})

            old_qso = old.get("qso_race_medal")
            new_qso = new.get("qso_race_medal")
            old_cf = old.get("cool_factor_medal")
            new_cf = new.get("cool_factor_medal")

            # Check if any medal changed
            if old_qso != new_qso or old_cf != new_cf:
                sport_name = new.get("sport_name") or old.get("sport_name", "Unknown")
                target = new.get("target_value") or old.get("target_value", "Unknown")
                points = new.get("total_points", 0)

                # Send push notification to the user
                notify_medal_change(
                    callsign=callsign,
                    sport_name=sport_name,
                    match_target=target,
                    old_medals={"qso_race": old_qso, "cool_factor": old_cf},
                    new_medals={"qso_race": new_qso, "cool_factor": new_cf},
                    total_points=points
                )

                # Collect newly won medals
                if new_qso and not old_qso:
                    changes.append({
                        "callsign": callsign,
                        "first_name": first_name,
                        "sport_name": sport_name,
                        "match_target": target,
                        "medal_type": new_qso,
                        "competition": "QSO Race",
                    })
                    if discord:
                        discord_notify_medal(callsign, sport_name, target, new_qso, "QSO Race",
                                             first_name=first_name)
                if new_cf and not old_cf:
                    changes.append({
                        "callsign": callsign,
                        "first_name": first_name,
                        "sport_name": sport_name,
                        "match_target": target,
                        "medal_type": new_cf,
                        "competition": "Cool Factor",
                    })
                    if discord:
                        discord_notify_medal(callsign, sport_name, target, new_cf, "Cool Factor",
                                             first_name=first_name)
    except Exception as e:
        logger.error(f"Failed to send medal notification for {callsign}: {e}")
    return changes
