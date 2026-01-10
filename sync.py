"""
Sync logic for fetching QSOs from QRZ and updating scores.
"""

from datetime import datetime
from typing import Optional

from database import get_db
from crypto import decrypt_api_key
from qrz_client import fetch_qsos, QSOData, QRZAPIError
from lotw_client import fetch_lotw_qsos, LoTWError
from grid_distance import grid_distance
from scoring import recompute_match_medals, update_records


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
        except Exception as e:
            return {"error": f"Failed to decrypt API key: {e}"}

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

    with get_db() as conn:
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
        except Exception as e:
            return {"error": f"Failed to decrypt LoTW credentials: {e}"}

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
    with get_db() as conn:
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
        # Check by callsign + datetime
        cursor = conn.execute(
            """SELECT id, is_confirmed FROM qsos
               WHERE competitor_callsign = ? AND dx_callsign = ?
               AND qso_datetime_utc = ?""",
            (competitor_callsign, qso.dx_callsign, qso.qso_datetime.isoformat())
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
        # Update if confirmation status changed
        if existing["is_confirmed"] != qso.is_confirmed:
            conn.execute("""
                UPDATE qsos SET
                    is_confirmed = ?,
                    band = ?,
                    mode = ?,
                    tx_power_w = ?,
                    my_dxcc = ?,
                    my_grid = ?,
                    my_sig_info = ?,
                    dx_dxcc = ?,
                    dx_grid = ?,
                    dx_sig_info = ?,
                    distance_km = ?,
                    cool_factor = ?
                WHERE id = ?
            """, (
                1 if qso.is_confirmed else 0,
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
        return None
    else:
        # Insert new QSO
        cursor = conn.execute("""
            INSERT INTO qsos (
                competitor_callsign, dx_callsign, qso_datetime_utc,
                band, mode, tx_power_w,
                my_dxcc, my_grid, my_sig_info,
                dx_dxcc, dx_grid, dx_sig_info,
                distance_km, cool_factor, is_confirmed, qrz_logid
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            qso.qrz_logid,
        ))

        # Update records for new confirmed QSOs
        if qso.is_confirmed and distance_km:
            update_records(cursor.lastrowid, competitor_callsign)

        return "new"


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
        # Get all matches in active Olympiad
        cursor = conn.execute("""
            SELECT m.id FROM matches m
            JOIN sports s ON m.sport_id = s.id
            JOIN olympiads o ON s.olympiad_id = o.id
            WHERE o.is_active = 1
        """)
        match_ids = [row["id"] for row in cursor.fetchall()]

    for match_id in match_ids:
        recompute_match_medals(match_id)


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
