"""
Sync logic for fetching QSOs from QRZ and updating scores.
"""

import binascii
import logging
from datetime import datetime
from typing import Optional

from cryptography.fernet import InvalidToken
from database import get_db, get_db_exclusive

logger = logging.getLogger(__name__)
from crypto import decrypt_api_key
from qrz_client import fetch_qsos, QSOData, QRZAPIError
from lotw_client import fetch_lotw_qsos, LoTWError
from grid_distance import grid_distance
from scoring import recompute_match_medals, compute_team_standings
from dxcc import get_continent_from_callsign


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


async def sync_competitor(callsign: str) -> dict:
    """
    Sync a single competitor's QSOs from QRZ using stored API key.

    Args:
        callsign: Competitor callsign

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
    return await sync_competitor_with_key(callsign, api_key)


async def sync_competitor_with_key(callsign: str, api_key: str) -> dict:
    """
    Sync a single competitor's QSOs from QRZ using provided API key.

    Args:
        callsign: Competitor callsign
        api_key: QRZ API key (not encrypted)

    Returns:
        dict with sync results
    """
    # Try to populate name if not set
    try:
        await populate_competitor_name(callsign)
    except Exception as e:
        logger.warning(f"Failed to populate name for {callsign}: {e}")

    # Fetch QSOs from QRZ
    try:
        qsos = await fetch_qsos(api_key, confirmed_only=False)
    except QRZAPIError as e:
        return {"error": str(e)}

    if not qsos:
        return {"message": "No QSOs found in QRZ logbook", "new_qsos": 0, "updated_qsos": 0}

    # Process and store QSOs
    new_count = 0
    updated_count = 0

    # Use exclusive transaction to prevent race conditions in upsert
    with get_db_exclusive() as conn:
        for qso in qsos:
            result = _upsert_qso(conn, callsign.upper(), qso)
            if result == "new":
                new_count += 1
            elif result == "updated":
                updated_count += 1

        # Update last sync time
        conn.execute(
            "UPDATE competitors SET last_sync_at = ? WHERE callsign = ?",
            (datetime.utcnow().isoformat(), callsign.upper())
        )

    # Recompute medals for all active matches after sync
    recompute_all_active_matches()

    return {
        "callsign": callsign.upper(),
        "new_qsos": new_count,
        "updated_qsos": updated_count,
        "total_fetched": len(qsos),
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

    # Use exclusive transaction to prevent race conditions in upsert
    with get_db_exclusive() as conn:
        # Verify competitor exists
        cursor = conn.execute(
            "SELECT * FROM competitors WHERE callsign = ?",
            (callsign.upper(),)
        )
        competitor = cursor.fetchone()

        if not competitor:
            return {"error": f"Competitor {callsign} not found"}

        # Fetch QSOs from LoTW
        try:
            qsos = await fetch_lotw_qsos(lotw_username, lotw_password, confirmed_only=False)
        except LoTWError as e:
            return {"error": str(e)}

        if not qsos:
            return {"message": "No QSOs found in LoTW", "new_qsos": 0, "updated_qsos": 0}

        # Process and store QSOs
        new_count = 0
        updated_count = 0

        for qso in qsos:
            result = _upsert_qso(conn, callsign.upper(), qso)
            if result == "new":
                new_count += 1
            elif result == "updated":
                updated_count += 1

        # Update last sync time
        conn.execute(
            "UPDATE competitors SET last_sync_at = ? WHERE callsign = ?",
            (datetime.utcnow().isoformat(), callsign.upper())
        )

    # Recompute medals for all active matches after sync
    recompute_all_active_matches()

    return {
        "callsign": callsign.upper(),
        "new_qsos": new_count,
        "updated_qsos": updated_count,
        "total_fetched": len(qsos),
    }


def _upsert_qso(conn, competitor_callsign: str, qso: QSOData) -> Optional[str]:
    """
    Insert or update a QSO in the database.

    Returns:
        "new", "updated", or None
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
        qso_dt = qso.qso_datetime.isoformat()
        cursor = conn.execute(
            """SELECT id, is_confirmed FROM qsos
               WHERE competitor_callsign = ? AND dx_callsign = ?
               AND (mode = ? OR mode IS NULL OR ? IS NULL)
               AND ABS(CAST((julianday(qso_datetime_utc) - julianday(?)) * 86400 AS INTEGER)) <= 120""",
            (competitor_callsign, qso.dx_callsign, qso.mode, qso.mode, qso_dt)
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
        return "updated"
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
            logger.info(f"Duplicate detected, merging: {competitor_callsign} -> {qso.dx_callsign} at {qso.qso_datetime}")
            cursor = conn.execute(
                """SELECT id FROM qsos
                   WHERE competitor_callsign = ? AND dx_callsign = ?
                   AND ABS(CAST((julianday(qso_datetime_utc) - julianday(?)) * 86400 AS INTEGER)) <= 120""",
                (competitor_callsign, qso.dx_callsign, qso.qso_datetime.isoformat())
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
                return "updated"
            return None

        # Note: Records are updated via recompute_all_records() which only
        # considers QSOs that qualified for matches (correct target, time period, mode)

        return "new"


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
            SELECT id, competitor_callsign, dx_callsign, mode, qso_datetime_utc
            FROM qsos
            ORDER BY competitor_callsign, dx_callsign, mode, qso_datetime_utc
        """)
        all_qsos = [dict(row) for row in cursor.fetchall()]

        # Find duplicates within 2-minute window (same callsigns AND same mode)
        duplicates_to_merge = []
        i = 0
        while i < len(all_qsos):
            qso = all_qsos[i]
            group = [qso["id"]]

            # Look ahead for QSOs with same callsigns AND same mode within 2 minutes
            j = i + 1
            while j < len(all_qsos):
                next_qso = all_qsos[j]
                if (next_qso["competitor_callsign"] != qso["competitor_callsign"] or
                    next_qso["dx_callsign"] != qso["dx_callsign"] or
                    next_qso["mode"] != qso["mode"]):
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

                # Delete the duplicate
                conn.execute("DELETE FROM qsos WHERE id = ?", (other["id"],))
                deleted_count += 1

            merged_count += 1

    return {
        "duplicate_groups": merged_count,
        "qsos_deleted": deleted_count,
    }


async def sync_all_competitors() -> dict:
    """
    Sync all competitors and recompute medals.

    Returns:
        dict with sync summary
    """
    with get_db() as conn:
        # Get all competitors
        cursor = conn.execute("SELECT callsign FROM competitors")
        callsigns = [row["callsign"] for row in cursor.fetchall()]

    results = []
    for callsign in callsigns:
        result = await sync_competitor(callsign)
        results.append(result)

    # Populate missing dx_dxcc values from callsign lookups
    try:
        dxcc_result = await populate_missing_dxcc()
        logger.info(f"Populated missing dx_dxcc: {dxcc_result}")
    except Exception as e:
        logger.warning(f"Failed to populate missing dx_dxcc: {e}")

    # Recompute medals for all active matches
    recompute_all_active_matches()

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

    # Recompute individual medals
    for match_id, _ in match_sport_pairs:
        recompute_match_medals(match_id)

    # Recompute team standings for each sport
    for sport_id in sport_ids:
        compute_team_standings(sport_id)
        # Sport-level done, now do match-level for matches in this sport
        sport_matches = [m for m, s in match_sport_pairs if s == sport_id]
        for match_id in sport_matches:
            compute_team_standings(sport_id, match_id)

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
        recompute_match_medals(match_id)

    # Recompute team standings for this sport
    compute_team_standings(sport_id)
    for match_id in match_ids:
        compute_team_standings(sport_id, match_id)

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
