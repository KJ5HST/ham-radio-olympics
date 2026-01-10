"""
Scoring engine for Ham Radio Olympics.

Handles:
- QSO matching against Sport/Match targets
- Medal computation for QSO Race and Cool Factor events
- POTA bonus logic
- Record tracking
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from database import get_db
from dxcc import get_continent
from grid_distance import grid_distance


@dataclass
class MatchingQSO:
    """A QSO that matches a specific Match target."""
    qso_id: int
    callsign: str
    dx_callsign: str
    qso_datetime: datetime
    distance_km: float
    tx_power: float
    cool_factor: float
    role: str  # 'work', 'activate', or 'combined'
    has_pota: bool  # competitor was at a park


@dataclass
class MedalResult:
    """Medal calculation result for a competitor."""
    callsign: str
    role: str
    qualified: bool
    qso_race_medal: Optional[str]
    qso_race_claim_time: Optional[datetime]
    cool_factor_medal: Optional[str]
    cool_factor_value: Optional[float]
    cool_factor_claim_time: Optional[datetime]
    pota_bonus: int  # 0, 1, or 2 (P2P)
    total_points: int


def matches_target(
    qso: dict,
    target_type: str,
    target_value: str,
    mode: str  # 'work' or 'activate'
) -> bool:
    """
    Check if a QSO matches a target for the given mode.

    Args:
        qso: QSO dictionary from database
        target_type: 'continent', 'country', 'park', 'call', 'grid'
        target_value: The target value to match
        mode: 'work' (check DX fields) or 'activate' (check MY_ fields)

    Returns:
        True if QSO matches target
    """
    target_value = target_value.upper().strip()

    if target_type == "continent":
        if mode == "work":
            dxcc = qso.get("dx_dxcc")
            if dxcc:
                continent = get_continent(dxcc)
                return continent == target_value
        else:  # activate
            dxcc = qso.get("my_dxcc")
            if dxcc:
                continent = get_continent(dxcc)
                return continent == target_value
        return False

    elif target_type == "country":
        if mode == "work":
            return str(qso.get("dx_dxcc", "")) == target_value
        else:
            return str(qso.get("my_dxcc", "")) == target_value

    elif target_type == "park":
        if mode == "work":
            sig_info = (qso.get("dx_sig_info") or "").upper().strip()
            return sig_info == target_value
        else:
            sig_info = (qso.get("my_sig_info") or "").upper().strip()
            return sig_info == target_value

    elif target_type == "call":
        if mode == "work":
            return (qso.get("dx_callsign") or "").upper() == target_value
        else:
            # For activate mode with call target, the competitor IS the callsign
            # This doesn't really make sense, but we check if they contacted someone
            return True  # All QSOs count when activating as a specific call

    elif target_type == "grid":
        if mode == "work":
            grid = (qso.get("dx_grid") or "").upper().strip()
            # Match if grid starts with target (e.g., target FN31 matches FN31ab)
            return grid.startswith(target_value)
        else:
            grid = (qso.get("my_grid") or "").upper().strip()
            return grid.startswith(target_value)

    return False


def validate_qso_for_mode(qso: dict, mode: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a QSO has required fields for the given mode.

    Args:
        qso: QSO dictionary
        mode: 'work' or 'activate'

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Always required: TX power must be present and positive
    tx_power = qso.get("tx_power_w")
    if tx_power is None or tx_power <= 0:
        return False, "TX power must be present and positive"

    # Must be confirmed
    if not qso.get("is_confirmed"):
        return False, "QSO not confirmed"

    if mode == "work":
        # Work mode requires DX station info
        if not qso.get("dx_dxcc"):
            return False, "Missing DXCC for work mode"
        if not qso.get("dx_grid"):
            return False, "Missing GRIDSQUARE for work mode"
    else:  # activate
        # Activate mode requires MY_ fields
        if not qso.get("my_grid"):
            return False, "Missing MY_GRIDSQUARE for activate mode"

    return True, None


def get_matching_qsos(
    match_id: int,
    sport_config: dict,
    target_value: str,
    start_date: datetime,
    end_date: datetime,
    sport_id: int = None,
) -> List[MatchingQSO]:
    """
    Find all QSOs that match a specific Match.

    Args:
        match_id: Match ID
        sport_config: Sport configuration dict
        target_value: Match target value
        start_date: Match start datetime
        end_date: Match end datetime
        sport_id: Sport ID (only competitors who opted in are included)

    Returns:
        List of MatchingQSO objects
    """
    matching = []

    target_type = sport_config["target_type"]
    work_enabled = sport_config["work_enabled"]
    activate_enabled = sport_config["activate_enabled"]
    separate_pools = sport_config["separate_pools"]

    with get_db() as conn:
        # Get all confirmed QSOs in the time window from competitors who opted in
        cursor = conn.execute("""
            SELECT q.*, c.registered_at
            FROM qsos q
            JOIN competitors c ON q.competitor_callsign = c.callsign
            JOIN sport_entries se ON q.competitor_callsign = se.callsign AND se.sport_id = ?
            WHERE q.is_confirmed = 1
            AND q.qso_datetime_utc >= ?
            AND q.qso_datetime_utc <= ?
        """, (sport_id, start_date.isoformat(), end_date.isoformat()))

        for row in cursor.fetchall():
            qso = dict(row)

            # Check work mode
            if work_enabled:
                valid, _ = validate_qso_for_mode(qso, "work")
                if valid and matches_target(qso, target_type, target_value, "work"):
                    role = "work" if separate_pools else "combined"
                    has_pota = bool(qso.get("my_sig_info"))

                    matching.append(MatchingQSO(
                        qso_id=qso["id"],
                        callsign=qso["competitor_callsign"],
                        dx_callsign=qso["dx_callsign"],
                        qso_datetime=datetime.fromisoformat(qso["qso_datetime_utc"]),
                        distance_km=qso["distance_km"] or 0,
                        tx_power=qso["tx_power_w"],
                        cool_factor=qso["cool_factor"] or 0,
                        role=role,
                        has_pota=has_pota,
                    ))

            # Check activate mode
            if activate_enabled:
                valid, _ = validate_qso_for_mode(qso, "activate")
                if valid and matches_target(qso, target_type, target_value, "activate"):
                    role = "activate" if separate_pools else "combined"
                    has_pota = bool(qso.get("my_sig_info"))

                    # Avoid duplicates if both modes enabled and not separate pools
                    if not (work_enabled and not separate_pools and
                            matches_target(qso, target_type, target_value, "work")):
                        matching.append(MatchingQSO(
                            qso_id=qso["id"],
                            callsign=qso["competitor_callsign"],
                            dx_callsign=qso["dx_callsign"],
                            qso_datetime=datetime.fromisoformat(qso["qso_datetime_utc"]),
                            distance_km=qso["distance_km"] or 0,
                            tx_power=qso["tx_power_w"],
                            cool_factor=qso["cool_factor"] or 0,
                            role=role,
                            has_pota=has_pota,
                        ))

    return matching


def compute_medals(
    matching_qsos: List[MatchingQSO],
    qualifying_qsos: int,
    target_type: str,
) -> List[MedalResult]:
    """
    Compute medals for a Match based on matching QSOs.

    Args:
        matching_qsos: List of QSOs matching the Match
        qualifying_qsos: Minimum QSOs to qualify for medals
        target_type: Sport's target type (for POTA bonus logic)

    Returns:
        List of MedalResult objects
    """
    if not matching_qsos:
        return []

    # Group QSOs by callsign and role
    by_competitor: Dict[Tuple[str, str], List[MatchingQSO]] = {}
    for qso in matching_qsos:
        key = (qso.callsign, qso.role)
        if key not in by_competitor:
            by_competitor[key] = []
        by_competitor[key].append(qso)

    results = []

    for (callsign, role), qsos in by_competitor.items():
        # Check qualification
        qualified = len(qsos) >= qualifying_qsos

        # QSO Race: first to make contact wins
        earliest_qso = min(qsos, key=lambda q: q.qso_datetime)
        qso_race_claim_time = earliest_qso.qso_datetime

        # Cool Factor event: highest cool factor
        best_cf_qso = max(qsos, key=lambda q: (q.cool_factor, -q.qso_datetime.timestamp()))
        cool_factor_value = best_cf_qso.cool_factor
        cool_factor_claim_time = best_cf_qso.qso_datetime

        # POTA bonus check
        has_pota = any(q.has_pota for q in qsos)
        pota_bonus = should_award_pota_bonus(target_type, role, has_pota)

        results.append(MedalResult(
            callsign=callsign,
            role=role,
            qualified=qualified,
            qso_race_medal=None,  # Computed below
            qso_race_claim_time=qso_race_claim_time,
            cool_factor_medal=None,  # Computed below
            cool_factor_value=cool_factor_value,
            cool_factor_claim_time=cool_factor_claim_time,
            pota_bonus=pota_bonus,
            total_points=0,  # Computed below
        ))

    # Award QSO Race medals (per role)
    for role in set(r.role for r in results):
        role_results = [r for r in results if r.role == role and r.qualified]
        # Sort by earliest claim time
        role_results.sort(key=lambda r: r.qso_race_claim_time)

        medals = ["gold", "silver", "bronze"]
        for i, result in enumerate(role_results[:3]):
            result.qso_race_medal = medals[i]

    # Award Cool Factor medals (per role)
    for role in set(r.role for r in results):
        role_results = [r for r in results if r.role == role and r.qualified]
        # Sort by highest cool factor, then earliest claim time
        role_results.sort(key=lambda r: (-r.cool_factor_value, r.cool_factor_claim_time))

        medals = ["gold", "silver", "bronze"]
        for i, result in enumerate(role_results[:3]):
            result.cool_factor_medal = medals[i]

    # Calculate total points
    medal_points = {"gold": 3, "silver": 2, "bronze": 1, None: 0}
    for result in results:
        points = medal_points[result.qso_race_medal]
        points += medal_points[result.cool_factor_medal]
        points += result.pota_bonus  # 0, 1, or 2 (P2P)
        result.total_points = points

    return results


def should_award_pota_bonus(target_type: str, role: str, competitor_at_park: bool) -> int:
    """
    Determine POTA bonus points to award.

    Logic:
    - Park-to-Park (POTA target + competitor at park): +2
    - POTA target OR competitor at park: +1
    - Neither: +0

    Args:
        target_type: Sport's target type
        role: 'work', 'activate', or 'combined'
        competitor_at_park: Whether competitor has MY_SIG_INFO

    Returns:
        Bonus points (0, 1, or 2)
    """
    is_pota_target = target_type == "park"

    if is_pota_target and competitor_at_park:
        # Park-to-Park: +2
        return 2
    elif is_pota_target or competitor_at_park:
        # Either POTA target or competitor at park: +1
        return 1
    else:
        # No POTA involvement
        return 0


def recompute_match_medals(match_id: int):
    """
    Recompute all medals for a Match.

    This should be called after a sync to update standings.
    """
    with get_db() as conn:
        # Get match and sport config
        cursor = conn.execute("""
            SELECT m.*, s.target_type, s.work_enabled, s.activate_enabled,
                   s.separate_pools, o.qualifying_qsos
            FROM matches m
            JOIN sports s ON m.sport_id = s.id
            JOIN olympiads o ON s.olympiad_id = o.id
            WHERE m.id = ?
        """, (match_id,))
        row = cursor.fetchone()
        if not row:
            return

        match_data = dict(row)
        start_date = datetime.fromisoformat(match_data["start_date"])
        end_date = datetime.fromisoformat(match_data["end_date"])

        sport_config = {
            "target_type": match_data["target_type"],
            "work_enabled": bool(match_data["work_enabled"]),
            "activate_enabled": bool(match_data["activate_enabled"]),
            "separate_pools": bool(match_data["separate_pools"]),
        }

        # Get matching QSOs (only from competitors who opted into this sport)
        matching = get_matching_qsos(
            match_id,
            sport_config,
            match_data["target_value"],
            start_date,
            end_date,
            sport_id=match_data["sport_id"],
        )

        # Compute medals
        results = compute_medals(
            matching,
            match_data["qualifying_qsos"],
            match_data["target_type"],
        )

        # Clear existing medals for this match
        conn.execute("DELETE FROM medals WHERE match_id = ?", (match_id,))

        # Insert new medals
        for result in results:
            conn.execute("""
                INSERT INTO medals (
                    match_id, callsign, role, qualified,
                    qso_race_medal, qso_race_claim_time,
                    cool_factor_medal, cool_factor_value, cool_factor_claim_time,
                    pota_bonus, total_points
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_id,
                result.callsign,
                result.role,
                1 if result.qualified else 0,
                result.qso_race_medal,
                result.qso_race_claim_time.isoformat() if result.qso_race_claim_time else None,
                result.cool_factor_medal,
                result.cool_factor_value,
                result.cool_factor_claim_time.isoformat() if result.cool_factor_claim_time else None,
                result.pota_bonus,
                result.total_points,
            ))


def update_records(qso_id: int, callsign: str, sport_id: Optional[int] = None):
    """
    Check and update records based on a QSO.

    Args:
        qso_id: QSO ID
        callsign: Competitor callsign
        sport_id: Sport ID (None for global records)
    """
    with get_db() as conn:
        # Get QSO data
        cursor = conn.execute("SELECT * FROM qsos WHERE id = ?", (qso_id,))
        qso = cursor.fetchone()
        if not qso:
            return

        qso = dict(qso)
        # Use QSO datetime for when record was achieved, not current time
        achieved_at = qso["qso_datetime_utc"]

        # Check longest distance
        if qso["distance_km"]:
            _update_record_if_better(
                conn, "longest_distance", qso["distance_km"],
                qso_id, callsign, sport_id, achieved_at, higher_is_better=True
            )

        # Check highest cool factor
        if qso["cool_factor"]:
            _update_record_if_better(
                conn, "highest_cool_factor", qso["cool_factor"],
                qso_id, callsign, sport_id, achieved_at, higher_is_better=True
            )

        # Check lowest power (only for confirmed QSOs with positive power)
        if qso["tx_power_w"] and qso["tx_power_w"] > 0 and qso["is_confirmed"]:
            _update_record_if_better(
                conn, "lowest_power", qso["tx_power_w"],
                qso_id, callsign, sport_id, achieved_at, higher_is_better=False
            )


def _update_record_if_better(
    conn,
    record_type: str,
    value: float,
    qso_id: int,
    callsign: str,
    sport_id: Optional[int],
    achieved_at: str,
    higher_is_better: bool,
):
    """Helper to update a record if the new value is better."""

    # Check world record (callsign=NULL, sport_id for sport-specific or NULL for global)
    cursor = conn.execute("""
        SELECT id, value FROM records
        WHERE record_type = ? AND callsign IS NULL AND sport_id IS ?
    """, (record_type, sport_id))
    world_record = cursor.fetchone()

    if world_record is None:
        # No record exists, create it
        conn.execute("""
            INSERT INTO records (sport_id, callsign, record_type, value, qso_id, achieved_at)
            VALUES (?, NULL, ?, ?, ?, ?)
        """, (sport_id, record_type, value, qso_id, achieved_at))
    else:
        is_better = (value > world_record["value"]) if higher_is_better else (value < world_record["value"])
        if is_better:
            conn.execute("""
                UPDATE records SET value = ?, qso_id = ?, achieved_at = ?
                WHERE id = ?
            """, (value, qso_id, achieved_at, world_record["id"]))

    # Check personal best (per callsign)
    cursor = conn.execute("""
        SELECT id, value FROM records
        WHERE record_type = ? AND callsign = ? AND sport_id IS ?
    """, (record_type, callsign, sport_id))
    pb_record = cursor.fetchone()

    if pb_record is None:
        conn.execute("""
            INSERT INTO records (sport_id, callsign, record_type, value, qso_id, achieved_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (sport_id, callsign, record_type, value, qso_id, achieved_at))
    else:
        is_better = (value > pb_record["value"]) if higher_is_better else (value < pb_record["value"])
        if is_better:
            conn.execute("""
                UPDATE records SET value = ?, qso_id = ?, achieved_at = ?
                WHERE id = ?
            """, (value, qso_id, achieved_at, pb_record["id"]))
