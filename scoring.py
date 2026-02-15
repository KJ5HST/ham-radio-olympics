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
from dxcc import get_continent, get_continent_from_callsign
from grid_distance import grid_distance


def parse_parks(sig_info: str) -> List[str]:
    """
    Parse park references from a SIG_INFO field.

    Handles comma-separated multiple parks (e.g., "US-0756,US-4568").
    Returns list of uppercase, stripped park references.
    """
    if not sig_info:
        return []
    parks = [p.strip().upper() for p in sig_info.split(',')]
    return [p for p in parks if p]  # Filter out empty strings


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
    is_confirmed: bool = True  # whether QSO is confirmed (False in live mode for unconfirmed)


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


def normalize_mode(mode: str) -> str:
    """
    Normalize QSO mode for comparison.

    Groups similar modes together:
    - USB, LSB -> SSB
    - FSK -> RTTY
    - Various digital modes stay as-is
    """
    if not mode:
        return ""
    mode = mode.upper().strip()
    # Normalize sideband modes to SSB
    if mode in ("USB", "LSB"):
        return "SSB"
    # Normalize FSK to RTTY
    if mode == "FSK":
        return "RTTY"
    return mode


def is_mode_allowed(qso_mode: str, allowed_modes: str) -> bool:
    """
    Check if a QSO's mode is in the allowed list.

    Args:
        qso_mode: The mode from the QSO record
        allowed_modes: Comma-separated list of allowed modes, or None/empty for all allowed

    Returns:
        True if mode is allowed
    """
    if not allowed_modes:
        return True  # No restriction, all modes allowed

    if not qso_mode:
        return False  # QSO has no mode, can't match restriction

    normalized_qso = normalize_mode(qso_mode)
    allowed_list = [normalize_mode(m.strip()) for m in allowed_modes.split(",") if m.strip()]

    if not allowed_list:
        return True  # Empty list means no restriction

    return normalized_qso in allowed_list


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
        target_type: 'continent', 'country', 'park', 'call', 'grid', 'any', 'pota'
        target_value: The target value to match
        mode: 'work' (check DX fields) or 'activate' (check MY_ fields)

    Returns:
        True if QSO matches target
    """
    target_value = target_value.upper().strip()

    # 'any' target type matches all QSOs (useful for general activity periods)
    if target_type == "any":
        return True

    # 'pota' target type: ANY park contact (not a specific park)
    # Work mode: DX station is at any park
    # Activate mode: Competitor is at any park
    if target_type == "pota":
        if mode == "work":
            sig_info = (qso.get("dx_sig_info") or "").strip()
            return bool(sig_info)  # True if DX is at any park
        else:
            sig_info = (qso.get("my_sig_info") or "").strip()
            return bool(sig_info)  # True if competitor is at any park

    if target_type == "continent":
        if mode == "work":
            dxcc = qso.get("dx_dxcc")
            if dxcc:
                # Trust DXCC code if available
                continent = get_continent(dxcc)
                return continent == target_value
            # Fallback: derive continent from callsign prefix only if no DXCC
            dx_call = qso.get("dx_callsign")
            if dx_call:
                continent = get_continent_from_callsign(dx_call)
                return continent == target_value
        else:  # activate
            dxcc = qso.get("my_dxcc")
            if dxcc:
                # Trust DXCC code if available
                continent = get_continent(dxcc)
                return continent == target_value
            # Fallback: derive continent from competitor's callsign only if no DXCC
            my_call = qso.get("competitor_callsign")
            if my_call:
                continent = get_continent_from_callsign(my_call)
                return continent == target_value
        return False

    elif target_type == "country":
        if mode == "work":
            return str(qso.get("dx_dxcc", "")) == target_value
        else:
            return str(qso.get("my_dxcc", "")) == target_value

    elif target_type == "park":
        # Parse multiple parks (comma-separated) and check if any match target
        if mode == "work":
            parks = parse_parks(qso.get("dx_sig_info") or "")
            return target_value in parks
        else:
            parks = parse_parks(qso.get("my_sig_info") or "")
            return target_value in parks

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


def validate_qso_for_mode(qso: dict, mode: str, live_mode: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validate that a QSO has required fields for the given mode.

    Note: TX power is NOT required for QSO validity. QSOs without valid power
    (missing, zero, or negative) are still valid for the QSO Race competition
    but are excluded from Cool Factor competition since cool_factor cannot be calculated.

    Args:
        qso: QSO dictionary
        mode: 'work' or 'activate'
        live_mode: If True, allow unconfirmed QSOs (for live results display)

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Must be confirmed (unless in live mode)
    if not live_mode and not qso.get("is_confirmed"):
        return False, "QSO not confirmed"

    if mode == "work":
        # Work mode requires DX station info
        # dx_dxcc is preferred, but we can fall back to deriving from callsign
        if not qso.get("dx_dxcc"):
            # Check if we can derive continent from callsign
            dx_call = qso.get("dx_callsign")
            if not dx_call or not get_continent_from_callsign(dx_call):
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
    match_allowed_modes: str = None,
    max_power_w: int = None,
    match_target_type: str = None,
    confirmation_deadline: datetime = None,
    live_mode: bool = False,
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
        match_allowed_modes: Match-level mode restriction (overrides sport if set)
        max_power_w: Maximum allowed TX power in watts (None means no limit)
        match_target_type: Match-level target_type override (None = use sport's)
        confirmation_deadline: QSOs confirmed after this time are excluded (None = no limit)
        live_mode: If True, include unconfirmed QSOs (for live results display)

    Returns:
        List of MatchingQSO objects
    """
    matching = []

    # Match-level target_type overrides sport-level if specified
    target_type = match_target_type if match_target_type else sport_config["target_type"]
    work_enabled = sport_config["work_enabled"]
    activate_enabled = sport_config["activate_enabled"]
    separate_pools = sport_config["separate_pools"]
    # Match-level modes override sport-level if specified
    allowed_modes = match_allowed_modes if match_allowed_modes else sport_config.get("allowed_modes")

    # POTA activation requires 10+ QSOs per day from the same park.
    # This is separate from the olympiad's qualifying_qsos (minimum to medal).
    # A valid POTA activation day = 10+ confirmed QSOs from that park on that UTC date.
    POTA_MIN_QSOS = 10

    with get_db() as conn:
        # Get QSOs in the time window from competitors who opted in
        # Exclude QSOs that are disqualified for this sport
        # In live_mode, include unconfirmed QSOs; otherwise require confirmation
        # If confirmation_deadline is set, only include QSOs confirmed before the deadline
        if confirmation_deadline:
            cursor = conn.execute("""
                SELECT q.*, c.registered_at
                FROM qsos q
                JOIN competitors c ON q.competitor_callsign = c.callsign
                JOIN sport_entries se ON q.competitor_callsign = se.callsign AND se.sport_id = ?
                LEFT JOIN qso_disqualifications dq
                    ON q.id = dq.qso_id AND dq.sport_id = ? AND dq.status = 'disqualified'
                WHERE (q.is_confirmed = 1 OR ? = 1)
                AND q.qso_datetime_utc >= ?
                AND q.qso_datetime_utc <= ?
                AND (q.confirmed_at IS NULL OR q.confirmed_at <= ?)
                AND dq.id IS NULL
            """, (sport_id, sport_id, 1 if live_mode else 0, start_date.isoformat(), end_date.isoformat(), confirmation_deadline.isoformat()))
        else:
            cursor = conn.execute("""
                SELECT q.*, c.registered_at
                FROM qsos q
                JOIN competitors c ON q.competitor_callsign = c.callsign
                JOIN sport_entries se ON q.competitor_callsign = se.callsign AND se.sport_id = ?
                LEFT JOIN qso_disqualifications dq
                    ON q.id = dq.qso_id AND dq.sport_id = ? AND dq.status = 'disqualified'
                WHERE (q.is_confirmed = 1 OR ? = 1)
                AND q.qso_datetime_utc >= ?
                AND q.qso_datetime_utc <= ?
                AND dq.id IS NULL
            """, (sport_id, sport_id, 1 if live_mode else 0, start_date.isoformat(), end_date.isoformat()))

        all_qsos = [dict(row) for row in cursor.fetchall()]

        # For park/pota activations, pre-compute valid activation days (10+ QSOs per day from same park)
        valid_activation_days = set()  # (callsign, park, date) tuples
        if activate_enabled and target_type in ("park", "pota"):
            # Count QSOs per (callsign, park, date)
            # Handle multiple parks per QSO - each park counted separately
            activation_counts: Dict[Tuple[str, str, str], int] = {}
            for qso in all_qsos:
                parks = parse_parks(qso.get("my_sig_info") or "")
                if not parks:
                    continue  # No park info
                callsign = qso["competitor_callsign"]
                qso_date = qso["qso_datetime_utc"][:10]  # UTC date
                for park in parks:
                    # For 'park' target: must match specific park
                    # For 'pota' target: any park counts
                    if target_type == "park" and park != target_value.upper().strip():
                        continue
                    key = (callsign, park, qso_date)
                    activation_counts[key] = activation_counts.get(key, 0) + 1

            # Only days with 10+ QSOs count as valid activations
            for key, count in activation_counts.items():
                if count >= POTA_MIN_QSOS:
                    valid_activation_days.add(key)

        for qso in all_qsos:
            # Check if QSO mode is allowed (applies to both work and activate)
            if not is_mode_allowed(qso.get("mode"), allowed_modes):
                continue

            # Check power limit (for QRP competitions)
            if max_power_w is not None:
                tx_power = qso.get("tx_power_w")
                if tx_power is None or tx_power > max_power_w:
                    continue

            # Check work mode
            if work_enabled:
                valid, _ = validate_qso_for_mode(qso, "work", live_mode=live_mode)
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
                        is_confirmed=bool(qso.get("is_confirmed")),
                    ))

            # Check activate mode
            if activate_enabled:
                valid, _ = validate_qso_for_mode(qso, "activate", live_mode=live_mode)
                if valid and matches_target(qso, target_type, target_value, "activate"):
                    # For park/pota activations, require 10+ QSOs on that day from that park
                    if target_type in ("park", "pota"):
                        parks = parse_parks(qso.get("my_sig_info") or "")
                        qso_date = qso["qso_datetime_utc"][:10]
                        callsign = qso["competitor_callsign"]
                        # Check if ANY of the parks has a valid activation day
                        has_valid_activation = any(
                            (callsign, park, qso_date) in valid_activation_days
                            for park in parks
                        )
                        if not has_valid_activation:
                            continue  # Skip - not a valid activation day for any park

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
                            is_confirmed=bool(qso.get("is_confirmed")),
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

        # Cool Factor event: highest cool factor (only QSOs with power/cool_factor)
        qsos_with_cf = [q for q in qsos if q.cool_factor and q.cool_factor > 0]
        if qsos_with_cf:
            best_cf_qso = max(qsos_with_cf, key=lambda q: (q.cool_factor, -q.qso_datetime.timestamp()))
            cool_factor_value = best_cf_qso.cool_factor
            cool_factor_claim_time = best_cf_qso.qso_datetime
        else:
            # No QSOs with power data - can't compete in Cool Factor
            cool_factor_value = 0
            cool_factor_claim_time = earliest_qso.qso_datetime

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

    # Award Cool Factor medals (per role) - only to those with valid cool factor
    for role in set(r.role for r in results):
        # Exclude competitors with no power data (cool_factor_value = 0)
        role_results = [r for r in results if r.role == role and r.qualified and r.cool_factor_value > 0]
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
    is_pota_target = target_type in ("park", "pota")

    if is_pota_target and competitor_at_park:
        # Park-to-Park: +2
        return 2
    elif is_pota_target or competitor_at_park:
        # Either POTA target or competitor at park: +1
        return 1
    else:
        # No POTA involvement
        return 0


def recompute_match_medals(match_id: int, notify: bool = True):
    """
    Recompute all medals for a Match.

    This should be called after a sync to update standings.
    With WAL mode enabled, we use regular transactions to allow concurrent reads.
    """
    # Collect QSOs for record updates (done outside transaction block to avoid deadlock)
    qsos_for_records = []

    with get_db() as conn:
        # Get match and sport config
        cursor = conn.execute("""
            SELECT m.*, s.target_type as sport_target_type, s.work_enabled, s.activate_enabled,
                   s.separate_pools, s.allowed_modes as sport_allowed_modes, o.qualifying_qsos
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
        confirmation_deadline = None
        if match_data.get("confirmation_deadline"):
            confirmation_deadline = datetime.fromisoformat(match_data["confirmation_deadline"])

        sport_config = {
            "target_type": match_data["sport_target_type"],
            "work_enabled": bool(match_data["work_enabled"]),
            "activate_enabled": bool(match_data["activate_enabled"]),
            "separate_pools": bool(match_data["separate_pools"]),
            "allowed_modes": match_data.get("sport_allowed_modes"),
        }

        # Get matching QSOs (only from competitors who opted into this sport)
        # Match-level allowed_modes and target_type override sport-level if set
        # live_mode includes unconfirmed QSOs for provisional standings
        live_mode = bool(match_data.get("show_live_results"))
        matching = get_matching_qsos(
            match_id,
            sport_config,
            match_data["target_value"],
            start_date,
            end_date,
            sport_id=match_data["sport_id"],
            match_allowed_modes=match_data.get("allowed_modes"),
            max_power_w=match_data.get("max_power_w"),
            match_target_type=match_data.get("target_type"),  # Match-level override
            confirmation_deadline=confirmation_deadline,
            live_mode=live_mode,
        )

        # Collect QSO info for record updates
        for qso in matching:
            qsos_for_records.append((qso.qso_id, qso.callsign))

        # Compute medals - use match target_type if set, otherwise sport's target_type
        effective_target_type = match_data.get("target_type") or match_data["sport_target_type"]
        results = compute_medals(
            matching,
            match_data["qualifying_qsos"],
            effective_target_type,
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

    # Update records after exclusive block is released
    for qso_id, callsign in qsos_for_records:
        update_records(qso_id, callsign, match_id=match_id, notify=notify)


def update_records(qso_id: int, callsign: str, sport_id: Optional[int] = None, match_id: Optional[int] = None, notify: bool = True):
    """
    Check and update records based on a QSO.

    With WAL mode enabled, we use regular transactions to allow concurrent reads.

    Args:
        qso_id: QSO ID
        callsign: Competitor callsign
        sport_id: Sport ID (None for global records)
        match_id: Match ID where the QSO qualified
        notify: Whether to send notifications for broken records (False during recomputes)
    """
    records_to_notify = []  # Collect records broken for notifications

    with get_db() as conn:
        # Get QSO data
        cursor = conn.execute("SELECT * FROM qsos WHERE id = ?", (qso_id,))
        qso = cursor.fetchone()
        if not qso:
            return

        qso = dict(qso)
        # Use QSO datetime for when record was achieved, not current time
        achieved_at = qso["qso_datetime_utc"]

        # Get sport name if available
        sport_name = None
        if sport_id:
            sport_row = conn.execute("SELECT name FROM sports WHERE id = ?", (sport_id,)).fetchone()
            if sport_row:
                sport_name = sport_row["name"]

        # Check longest distance
        if qso["distance_km"]:
            result = _update_record_if_better(
                conn, "longest_distance", qso["distance_km"],
                qso_id, callsign, sport_id, match_id, achieved_at, higher_is_better=True
            )
            if result["world_record_broken"] or result["personal_best_broken"]:
                records_to_notify.append({
                    "record_type": "distance",
                    "value": qso["distance_km"],
                    "is_world_record": result["world_record_broken"],
                    "sport_name": sport_name,
                    "dx_callsign": qso["dx_callsign"],
                })

        # Check highest cool factor
        if qso["cool_factor"]:
            result = _update_record_if_better(
                conn, "highest_cool_factor", qso["cool_factor"],
                qso_id, callsign, sport_id, match_id, achieved_at, higher_is_better=True
            )
            if result["world_record_broken"] or result["personal_best_broken"]:
                records_to_notify.append({
                    "record_type": "cool_factor",
                    "value": qso["cool_factor"],
                    "is_world_record": result["world_record_broken"],
                    "sport_name": sport_name,
                    "dx_callsign": qso["dx_callsign"],
                })

        # Check lowest power (only for confirmed QSOs with positive power)
        if qso["tx_power_w"] and qso["tx_power_w"] > 0 and qso["is_confirmed"]:
            result = _update_record_if_better(
                conn, "lowest_power", qso["tx_power_w"],
                qso_id, callsign, sport_id, match_id, achieved_at, higher_is_better=False
            )
            # Don't notify for lowest power records as they're less exciting

    # Send notifications after transaction commits (skip during recomputes)
    if notify:
        _notify_records_broken(callsign, records_to_notify)


def _notify_records_broken(callsign: str, records: list):
    """Send push notifications for broken records."""
    if not records:
        return

    try:
        from notifications import notify_record_broken, discord_notify_record
        from database import get_db

        # Look up first name for Discord display
        with get_db() as conn:
            row = conn.execute("SELECT first_name FROM competitors WHERE callsign = ?",
                               (callsign,)).fetchone()
            first_name = row["first_name"] if row else None

        for record in records:
            # Send push notification to the user
            notify_record_broken(
                callsign=callsign,
                record_type=record["record_type"],
                value=record["value"],
                is_world_record=record["is_world_record"],
                sport_name=record.get("sport_name"),
                dx_callsign=record.get("dx_callsign"),
            )
            # Send Discord notification (only for world records)
            discord_notify_record(
                callsign=callsign,
                record_type=record["record_type"],
                value=record["value"],
                is_world_record=record["is_world_record"],
                sport_name=record.get("sport_name"),
                dx_callsign=record.get("dx_callsign"),
                first_name=first_name,
            )
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to send record notification for {callsign}: {e}")


def _update_record_if_better(
    conn,
    record_type: str,
    value: float,
    qso_id: int,
    callsign: str,
    sport_id: Optional[int],
    match_id: Optional[int],
    achieved_at: str,
    higher_is_better: bool,
) -> dict:
    """
    Helper to update a record if the new value is better.

    Returns dict with:
        - world_record_broken: True if a world record was broken
        - personal_best_broken: True if a personal best was broken
    """
    result = {"world_record_broken": False, "personal_best_broken": False}

    # Build sport_id condition - use IS NULL for NULL, = for values
    if sport_id is None:
        sport_condition = "sport_id IS NULL"
        sport_params = ()
    else:
        sport_condition = "sport_id = ?"
        sport_params = (sport_id,)

    # Check world record (callsign=NULL, sport_id for sport-specific or NULL for global)
    cursor = conn.execute(f"""
        SELECT id, value FROM records
        WHERE record_type = ? AND callsign IS NULL AND {sport_condition}
    """, (record_type,) + sport_params)
    world_record = cursor.fetchone()

    if world_record is None:
        # No record exists, create it (first record = world record)
        conn.execute("""
            INSERT INTO records (sport_id, match_id, callsign, record_type, value, qso_id, achieved_at)
            VALUES (?, ?, NULL, ?, ?, ?, ?)
        """, (sport_id, match_id, record_type, value, qso_id, achieved_at))
        result["world_record_broken"] = True
    else:
        is_better = (value > world_record["value"]) if higher_is_better else (value < world_record["value"])
        if is_better:
            conn.execute("""
                UPDATE records SET value = ?, qso_id = ?, match_id = ?, achieved_at = ?
                WHERE id = ?
            """, (value, qso_id, match_id, achieved_at, world_record["id"]))
            result["world_record_broken"] = True

    # Check personal best (per callsign)
    cursor = conn.execute(f"""
        SELECT id, value FROM records
        WHERE record_type = ? AND callsign = ? AND {sport_condition}
    """, (record_type, callsign) + sport_params)
    pb_record = cursor.fetchone()

    if pb_record is None:
        conn.execute("""
            INSERT INTO records (sport_id, match_id, callsign, record_type, value, qso_id, achieved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (sport_id, match_id, callsign, record_type, value, qso_id, achieved_at))
        result["personal_best_broken"] = True
    else:
        is_better = (value > pb_record["value"]) if higher_is_better else (value < pb_record["value"])
        if is_better:
            conn.execute("""
                UPDATE records SET value = ?, qso_id = ?, match_id = ?, achieved_at = ?
                WHERE id = ?
            """, (value, qso_id, match_id, achieved_at, pb_record["id"]))
            result["personal_best_broken"] = True

    return result


@dataclass
class TeamStanding:
    """Team standing result for a sport or match."""
    team_id: int
    team_name: str
    total_points: float
    member_count: int
    gold_count: int
    silver_count: int
    bronze_count: int
    medal: Optional[str]  # Team medal for this calculation


def compute_team_standings(
    sport_id: int,
    match_id: Optional[int] = None,
    top_n: int = 3
):
    """
    Compute team standings for a sport or specific match.

    Uses four calculation methods:
    - normalized: Sum of top N members' points, where N = smallest team size
    - top_n: Sum of top 3 members' points
    - average: Total points / member count
    - sum: Raw sum of all members' points

    Args:
        sport_id: Sport ID to compute standings for
        match_id: Optional match ID (None for sport-level standings)
        top_n: Number of top members for top_n method (default 3)

    With WAL mode enabled, we use regular transactions to allow concurrent reads.
    """
    with get_db() as conn:
        # Get all active teams with members
        cursor = conn.execute("""
            SELECT t.id, t.name, tm.callsign
            FROM teams t
            JOIN team_members tm ON t.id = tm.team_id
            WHERE t.is_active = 1
            ORDER BY t.id, tm.callsign
        """)

        # Build team membership map
        teams: Dict[int, dict] = {}
        for row in cursor.fetchall():
            team_id = row["id"]
            if team_id not in teams:
                teams[team_id] = {
                    "id": team_id,
                    "name": row["name"],
                    "members": []
                }
            teams[team_id]["members"].append(row["callsign"])

        if not teams:
            return  # No teams to compute

        # Build medal query based on match_id
        if match_id:
            # Specific match
            medal_query = """
                SELECT callsign,
                       COALESCE(SUM(total_points), 0) as points,
                       COUNT(CASE WHEN qso_race_medal = 'gold' OR cool_factor_medal = 'gold' THEN 1 END) as gold,
                       COUNT(CASE WHEN qso_race_medal = 'silver' OR cool_factor_medal = 'silver' THEN 1 END) as silver,
                       COUNT(CASE WHEN qso_race_medal = 'bronze' OR cool_factor_medal = 'bronze' THEN 1 END) as bronze
                FROM medals
                WHERE match_id = ?
                GROUP BY callsign
            """
            cursor = conn.execute(medal_query, (match_id,))
        else:
            # All matches in sport
            medal_query = """
                SELECT m.callsign,
                       COALESCE(SUM(m.total_points), 0) as points,
                       COUNT(CASE WHEN m.qso_race_medal = 'gold' OR m.cool_factor_medal = 'gold' THEN 1 END) as gold,
                       COUNT(CASE WHEN m.qso_race_medal = 'silver' OR m.cool_factor_medal = 'silver' THEN 1 END) as silver,
                       COUNT(CASE WHEN m.qso_race_medal = 'bronze' OR m.cool_factor_medal = 'bronze' THEN 1 END) as bronze
                FROM medals m
                JOIN matches ma ON m.match_id = ma.id
                WHERE ma.sport_id = ?
                GROUP BY m.callsign
            """
            cursor = conn.execute(medal_query, (sport_id,))

        # Build member points map
        member_points: Dict[str, dict] = {}
        for row in cursor.fetchall():
            member_points[row["callsign"]] = {
                "points": row["points"],
                "gold": row["gold"],
                "silver": row["silver"],
                "bronze": row["bronze"]
            }

        # Compute team stats for each method
        # Find smallest team size (for normalized calculation)
        min_team_size = min(len(t["members"]) for t in teams.values())

        # Clear existing team medals for this sport/match
        if match_id:
            conn.execute(
                "DELETE FROM team_medals WHERE sport_id = ? AND match_id = ?",
                (sport_id, match_id)
            )
        else:
            conn.execute(
                "DELETE FROM team_medals WHERE sport_id = ? AND match_id IS NULL",
                (sport_id,)
            )

        methods = ["normalized", "top_n", "average", "sum"]

        for method in methods:
            standings: List[TeamStanding] = []

            for team_id, team_data in teams.items():
                members = team_data["members"]
                # Get points for each member, sorted descending
                member_scores = []
                total_gold = 0
                total_silver = 0
                total_bronze = 0

                for callsign in members:
                    stats = member_points.get(callsign, {"points": 0, "gold": 0, "silver": 0, "bronze": 0})
                    member_scores.append(stats["points"])
                    total_gold += stats["gold"]
                    total_silver += stats["silver"]
                    total_bronze += stats["bronze"]

                member_scores.sort(reverse=True)

                # Calculate team score based on method
                if method == "normalized":
                    # Sum top N where N = smallest team size
                    total = sum(member_scores[:min_team_size])
                elif method == "top_n":
                    # Sum top 3 (or fewer if team is smaller)
                    total = sum(member_scores[:top_n])
                elif method == "average":
                    # Average of all members
                    total = sum(member_scores) / len(member_scores) if member_scores else 0
                else:  # sum
                    # Raw sum of all
                    total = sum(member_scores)

                standings.append(TeamStanding(
                    team_id=team_id,
                    team_name=team_data["name"],
                    total_points=total,
                    member_count=len(members),
                    gold_count=total_gold,
                    silver_count=total_silver,
                    bronze_count=total_bronze,
                    medal=None
                ))

            # Sort by total points descending
            standings.sort(key=lambda s: (-s.total_points, s.team_name))

            # Award team medals (top 3)
            medals = ["gold", "silver", "bronze"]
            for i, standing in enumerate(standings[:3]):
                if standing.total_points > 0:  # Only award if they have points
                    standing.medal = medals[i]

            # Store results
            now = datetime.utcnow().isoformat()
            for standing in standings:
                conn.execute("""
                    INSERT INTO team_medals (
                        team_id, match_id, sport_id, calculation_method,
                        total_points, member_count, gold_count, silver_count, bronze_count,
                        medal, computed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    standing.team_id,
                    match_id,
                    sport_id,
                    method,
                    standing.total_points,
                    standing.member_count,
                    standing.gold_count,
                    standing.silver_count,
                    standing.bronze_count,
                    standing.medal,
                    now
                ))


def recompute_all_team_standings():
    """
    Recompute team standings for all sports and matches.

    Call this after medal recomputation to update team rankings.
    Skips recomputation entirely if there are no matches.
    """
    with get_db() as conn:
        # Skip if there are no matches at all
        match_count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        if match_count == 0:
            return

        # Get all sports
        cursor = conn.execute("SELECT id FROM sports")
        sport_ids = [row["id"] for row in cursor.fetchall()]

    for sport_id in sport_ids:
        # Compute match-level standings
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT id FROM matches WHERE sport_id = ?",
                (sport_id,)
            )
            match_ids = [row["id"] for row in cursor.fetchall()]

        if not match_ids:
            continue

        # Compute sport-level standings
        compute_team_standings(sport_id)

        for match_id in match_ids:
            compute_team_standings(sport_id, match_id)


def recompute_all_records():
    """
    Recompute all records from scratch.

    Records are only credited to matches where the QSO actually matches the target.
    For example, a QSO with VK6AS (Oceania) won't be credited to an NA (North America) match.
    """
    with get_db() as conn:
        # Clear all existing records
        conn.execute("DELETE FROM records")

        # Find all confirmed QSOs from medal-holding competitors during match periods
        # Include match and sport info for target validation
        cursor = conn.execute("""
            WITH medal_competitor_matches AS (
                -- Get all competitor/match pairs that have medals
                SELECT DISTINCT med.callsign, med.match_id
                FROM medals med
                WHERE med.qualified = 1
            )
            SELECT DISTINCT q.id as qso_id, q.competitor_callsign, m.id as match_id,
                   s.target_type, COALESCE(m.target_type, s.target_type) as effective_target_type,
                   m.target_value, s.work_enabled, s.activate_enabled
            FROM medal_competitor_matches mcm
            INNER JOIN matches m ON mcm.match_id = m.id
            INNER JOIN sports s ON m.sport_id = s.id
            INNER JOIN qsos q ON q.competitor_callsign = mcm.callsign
                             AND q.qso_datetime_utc >= m.start_date
                             AND q.qso_datetime_utc <= m.end_date
                             AND q.is_confirmed = 1
            WHERE q.distance_km IS NOT NULL OR q.cool_factor IS NOT NULL
        """)
        qsos_to_process = cursor.fetchall()

        # Build a dict of QSO data for target matching
        qso_cache = {}
        for row in qsos_to_process:
            qso_id = row["qso_id"]
            if qso_id not in qso_cache:
                qso_row = conn.execute("SELECT * FROM qsos WHERE id = ?", (qso_id,)).fetchone()
                if qso_row:
                    qso_cache[qso_id] = dict(qso_row)

    # Track (qso_id, match_id) pairs we've processed to avoid duplicates
    seen_pairs = set()

    for row in qsos_to_process:
        qso_id = row["qso_id"]
        callsign = row["competitor_callsign"]
        match_id = row["match_id"]
        target_type = row["effective_target_type"]
        target_value = row["target_value"]
        work_enabled = row["work_enabled"]
        activate_enabled = row["activate_enabled"]

        pair_key = (qso_id, match_id)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        # Get the QSO data for target matching
        qso = qso_cache.get(qso_id)
        if not qso:
            continue

        # Check if QSO actually matches the target for this match
        qso_matches = False
        if work_enabled and matches_target(qso, target_type, target_value, "work"):
            qso_matches = True
        if activate_enabled and matches_target(qso, target_type, target_value, "activate"):
            qso_matches = True

        if qso_matches:
            update_records(qso_id, callsign, match_id=match_id, notify=False)


def get_honorable_mentions() -> dict:
    """
    Find confirmed QSOs that beat current records but weren't during competition.

    An honorable mention is awarded when a confirmed QSO has a better value than
    the current world record, but the QSO wasn't made during competition (either
    outside the olympiad date range or inside but doesn't match any match target).

    Returns dict with keys:
    - 'longest_distance': {callsign, value, date, qso_id, first_name} or None
    - 'highest_cool_factor': {callsign, value, date, qso_id, first_name} or None
    """
    result = {
        'longest_distance': None,
        'highest_cool_factor': None,
    }

    with get_db() as conn:
        # Get current world records
        cursor = conn.execute("""
            SELECT record_type, value, qso_id
            FROM records
            WHERE callsign IS NULL AND sport_id IS NULL
            AND record_type IN ('longest_distance', 'highest_cool_factor')
        """)
        world_records = {row['record_type']: row for row in cursor.fetchall()}

        # Get active olympiad date range
        cursor = conn.execute("""
            SELECT start_date, end_date FROM olympiads WHERE is_active = 1
        """)
        olympiad = cursor.fetchone()
        olympiad_start = olympiad['start_date'] if olympiad else None
        olympiad_end = olympiad['end_date'] if olympiad else None

        # Get all match date ranges and their target info for the active olympiad
        competition_qso_ids = set()
        if olympiad:
            cursor = conn.execute("""
                SELECT m.id, m.start_date, m.end_date, m.target_value,
                       COALESCE(m.target_type, s.target_type) as target_type,
                       s.work_enabled, s.activate_enabled
                FROM matches m
                JOIN sports s ON m.sport_id = s.id
                WHERE s.olympiad_id = (SELECT id FROM olympiads WHERE is_active = 1)
            """)
            matches = [dict(row) for row in cursor.fetchall()]

            # Find all QSO IDs that matched any competition target
            for match in matches:
                cursor = conn.execute("""
                    SELECT q.id, q.dx_dxcc, q.dx_grid, q.dx_sig_info,
                           q.my_dxcc, q.my_grid, q.my_sig_info, q.dx_callsign,
                           q.competitor_callsign
                    FROM qsos q
                    WHERE q.is_confirmed = 1
                    AND q.qso_datetime_utc >= ?
                    AND q.qso_datetime_utc <= ?
                """, (match['start_date'], match['end_date']))

                for qso in cursor.fetchall():
                    qso_dict = dict(qso)
                    # Check if this QSO matches the match target
                    qso_matches = False
                    if match['work_enabled'] and matches_target(
                        qso_dict, match['target_type'], match['target_value'], "work"
                    ):
                        qso_matches = True
                    if match['activate_enabled'] and matches_target(
                        qso_dict, match['target_type'], match['target_value'], "activate"
                    ):
                        qso_matches = True

                    if qso_matches:
                        competition_qso_ids.add(qso_dict['id'])

        # Find best overall confirmed QSOs (excluding competition QSOs)
        # For longest distance
        cursor = conn.execute("""
            SELECT q.id, q.competitor_callsign, q.distance_km, q.qso_datetime_utc,
                   c.first_name
            FROM qsos q
            JOIN competitors c ON q.competitor_callsign = c.callsign
            WHERE q.is_confirmed = 1
            AND q.distance_km IS NOT NULL
            AND q.distance_km > 0
            ORDER BY q.distance_km DESC
            LIMIT 100
        """)
        distance_candidates = [dict(row) for row in cursor.fetchall()]

        # Find the best non-competition QSO
        current_distance_record = world_records.get('longest_distance')
        for qso in distance_candidates:
            if qso['id'] not in competition_qso_ids:
                # Only an honorable mention if there IS a record and this QSO beats it
                if current_distance_record and qso['distance_km'] > current_distance_record['value']:
                    result['longest_distance'] = {
                        'callsign': qso['competitor_callsign'],
                        'value': qso['distance_km'],
                        'date': qso['qso_datetime_utc'],
                        'qso_id': qso['id'],
                        'first_name': qso['first_name'],
                    }
                break  # Only consider the best non-competition QSO

        # For highest cool factor
        cursor = conn.execute("""
            SELECT q.id, q.competitor_callsign, q.cool_factor, q.qso_datetime_utc,
                   c.first_name
            FROM qsos q
            JOIN competitors c ON q.competitor_callsign = c.callsign
            WHERE q.is_confirmed = 1
            AND q.cool_factor IS NOT NULL
            AND q.cool_factor > 0
            ORDER BY q.cool_factor DESC
            LIMIT 100
        """)
        cf_candidates = [dict(row) for row in cursor.fetchall()]

        current_cf_record = world_records.get('highest_cool_factor')
        for qso in cf_candidates:
            if qso['id'] not in competition_qso_ids:
                # Only an honorable mention if there IS a record and this QSO beats it
                if current_cf_record and qso['cool_factor'] > current_cf_record['value']:
                    result['highest_cool_factor'] = {
                        'callsign': qso['competitor_callsign'],
                        'value': qso['cool_factor'],
                        'date': qso['qso_datetime_utc'],
                        'qso_id': qso['id'],
                        'first_name': qso['first_name'],
                    }
                break

    return result


def compute_triathlon_leaders(limit: int = 3) -> List[dict]:
    """
    Compute top Triathlon QSOs across the active olympiad.

    Triathlon Score = Distance Percentile (0-100) + Cool Factor Percentile (0-100) + POTA Bonus (50 or 100)

    The three events:
    - Distance: How far the QSO traveled
    - Cool Factor: Efficiency (distance/power ratio)
    - POTA: Park involvement bonus

    Requirements:
    - distance_km > 0
    - tx_power_w > 0 (for cool_factor)
    - POTA involvement (my_sig_info OR dx_sig_info present)
    - P2P (both parks) = 100, single park = 50

    Args:
        limit: Maximum number of leaders to return (default 3)

    Returns:
        List of dicts with QSO details and score breakdown
    """
    with get_db() as conn:
        # Get the active olympiad
        cursor = conn.execute("""
            SELECT id, start_date, end_date FROM olympiads WHERE is_active = 1
        """)
        olympiad = cursor.fetchone()
        if not olympiad:
            return []

        # Get all confirmed QSOs from the active olympiad period that qualify:
        # - distance_km > 0
        # - tx_power_w > 0
        # - POTA involvement (my_sig_info OR dx_sig_info present)
        cursor = conn.execute("""
            SELECT q.id, q.competitor_callsign, q.dx_callsign, q.qso_datetime_utc,
                   q.distance_km, q.tx_power_w, q.cool_factor,
                   q.my_sig_info, q.dx_sig_info, q.mode, q.band,
                   c.first_name
            FROM qsos q
            JOIN competitors c ON q.competitor_callsign = c.callsign
            WHERE q.is_confirmed = 1
              AND q.qso_datetime_utc >= ?
              AND q.qso_datetime_utc <= ?
              AND q.distance_km > 0
              AND q.tx_power_w > 0
              AND (q.my_sig_info IS NOT NULL AND q.my_sig_info != ''
                   OR q.dx_sig_info IS NOT NULL AND q.dx_sig_info != '')
            ORDER BY q.distance_km DESC
        """, (olympiad["start_date"], olympiad["end_date"]))

        qualifying_qsos = [dict(row) for row in cursor.fetchall()]

        if not qualifying_qsos:
            return []

        # Build sorted lists for percentile calculation
        distances = sorted([q["distance_km"] for q in qualifying_qsos])
        cool_factors = sorted([q["cool_factor"] for q in qualifying_qsos])

        # Calculate percentile for each QSO
        def percentile_rank(value: float, sorted_list: List[float]) -> float:
            """Calculate percentile rank (0-100) for a value in a sorted list."""
            if not sorted_list:
                return 0
            # Count values less than or equal to this value
            count_le = sum(1 for v in sorted_list if v <= value)
            # Percentile = (count of values <= this value) / total count * 100
            return (count_le / len(sorted_list)) * 100

        scored_qsos = []
        for qso in qualifying_qsos:
            # Calculate distance percentile
            distance_pct = percentile_rank(qso["distance_km"], distances)

            # Calculate cool factor percentile
            cf_pct = percentile_rank(qso["cool_factor"], cool_factors)

            # Calculate POTA bonus
            has_my_park = bool(qso["my_sig_info"])
            has_dx_park = bool(qso["dx_sig_info"])
            if has_my_park and has_dx_park:
                pota_bonus = 100  # P2P
            else:
                pota_bonus = 50  # Single park

            # Total triathlon score
            total_score = distance_pct + cf_pct + pota_bonus

            scored_qsos.append({
                "qso_id": qso["id"],
                "callsign": qso["competitor_callsign"],
                "first_name": qso["first_name"],
                "dx_callsign": qso["dx_callsign"],
                "qso_datetime": qso["qso_datetime_utc"],
                "distance_km": qso["distance_km"],
                "tx_power_w": qso["tx_power_w"],
                "cool_factor": qso["cool_factor"],
                "mode": qso["mode"],
                "band": qso["band"],
                "my_sig_info": qso["my_sig_info"],
                "dx_sig_info": qso["dx_sig_info"],
                "distance_percentile": distance_pct,
                "cool_factor_percentile": cf_pct,
                "pota_bonus": pota_bonus,
                "total_score": total_score,
            })

        # Sort by total score (desc), then QSO timestamp (asc) for ties
        scored_qsos.sort(key=lambda q: (-q["total_score"], q["qso_datetime"]))

        # Each competitor can only appear once on the podium (best QSO only)
        seen_callsigns = set()
        unique_leaders = []
        for qso in scored_qsos:
            if qso["callsign"] not in seen_callsigns:
                seen_callsigns.add(qso["callsign"])
                unique_leaders.append(qso)
                if len(unique_leaders) >= limit:
                    break

        return unique_leaders


def compute_mode_records() -> Tuple[List[dict], List[dict]]:
    """
    Compute distance and cool factor records by mode.

    Returns:
        Tuple of (distance_records, cool_factor_records)
        Each record contains: mode, value, holder, holder_name, sport_id,
        match_id, sport_name, target, dx_callsign, power, cool_factor/distance,
        date, my_sig_info
    """
    distance_records = []
    cool_factor_records = []

    with get_db() as conn:
        # Get all global records that have a qso_id (actual records, not placeholders)
        cursor = conn.execute("""
            SELECT r.record_type, r.value, r.qso_id, r.match_id,
                   q.competitor_callsign as holder, q.dx_callsign, q.mode, q.band,
                   q.tx_power_w, q.distance_km, q.cool_factor, q.qso_datetime_utc,
                   q.my_sig_info,
                   c.first_name as holder_name,
                   m.target_value as target,
                   s.id as sport_id, s.name as sport_name
            FROM records r
            JOIN qsos q ON r.qso_id = q.id
            JOIN competitors c ON q.competitor_callsign = c.callsign
            LEFT JOIN matches m ON r.match_id = m.id
            LEFT JOIN sports s ON m.sport_id = s.id
            WHERE r.callsign IS NULL AND r.sport_id IS NULL
            AND q.mode IS NOT NULL
        """)
        world_records = [dict(row) for row in cursor.fetchall()]

        # Get distinct modes from confirmed QSOs
        cursor = conn.execute("""
            SELECT DISTINCT mode FROM qsos
            WHERE is_confirmed = 1 AND mode IS NOT NULL AND mode != ''
        """)
        modes = [row["mode"] for row in cursor.fetchall()]

        # For each mode, find the best distance and cool factor records
        # Only include QSOs that qualified for medals (made during competition)
        # Join through medals to find competitors who medaled, then find their
        # QSOs that fall within the match date range
        for mode in modes:
            # Best distance for this mode (only from medal-qualifying QSOs)
            cursor = conn.execute("""
                SELECT q.id, q.competitor_callsign as holder, q.dx_callsign,
                       q.distance_km as value, q.tx_power_w as power, q.cool_factor,
                       q.qso_datetime_utc as date, q.my_sig_info,
                       c.first_name as holder_name,
                       m.target_value as target, m.id as match_id,
                       s.id as sport_id, s.name as sport_name
                FROM medals med
                JOIN matches m ON med.match_id = m.id
                JOIN sports s ON m.sport_id = s.id
                JOIN qsos q ON q.competitor_callsign = med.callsign
                    AND q.qso_datetime_utc >= m.start_date
                    AND q.qso_datetime_utc <= m.end_date || ' 23:59:59'
                    AND q.is_confirmed = 1
                JOIN competitors c ON q.competitor_callsign = c.callsign
                WHERE med.qualified = 1
                AND q.mode = ?
                AND q.distance_km IS NOT NULL
                AND q.distance_km > 0
                ORDER BY q.distance_km DESC
                LIMIT 1
            """, (mode,))
            row = cursor.fetchone()
            if row:
                distance_records.append({
                    "mode": mode,
                    "value": row["value"],
                    "holder": row["holder"],
                    "holder_name": row["holder_name"],
                    "sport_id": row["sport_id"],
                    "match_id": row["match_id"],
                    "sport_name": row["sport_name"],
                    "target": row["target"],
                    "dx_callsign": row["dx_callsign"],
                    "power": row["power"],
                    "cool_factor": row["cool_factor"],
                    "date": row["date"],
                    "my_sig_info": row["my_sig_info"],
                })

            # Best cool factor for this mode (only from medal-qualifying QSOs)
            cursor = conn.execute("""
                SELECT q.id, q.competitor_callsign as holder, q.dx_callsign,
                       q.cool_factor as value, q.tx_power_w as power, q.distance_km as distance,
                       q.qso_datetime_utc as date, q.my_sig_info,
                       c.first_name as holder_name,
                       m.target_value as target, m.id as match_id,
                       s.id as sport_id, s.name as sport_name
                FROM medals med
                JOIN matches m ON med.match_id = m.id
                JOIN sports s ON m.sport_id = s.id
                JOIN qsos q ON q.competitor_callsign = med.callsign
                    AND q.qso_datetime_utc >= m.start_date
                    AND q.qso_datetime_utc <= m.end_date || ' 23:59:59'
                    AND q.is_confirmed = 1
                JOIN competitors c ON q.competitor_callsign = c.callsign
                WHERE med.qualified = 1
                AND q.mode = ?
                AND q.cool_factor IS NOT NULL
                AND q.cool_factor > 0
                ORDER BY q.cool_factor DESC
                LIMIT 1
            """, (mode,))
            row = cursor.fetchone()
            if row:
                cool_factor_records.append({
                    "mode": mode,
                    "value": row["value"],
                    "holder": row["holder"],
                    "holder_name": row["holder_name"],
                    "sport_id": row["sport_id"],
                    "match_id": row["match_id"],
                    "sport_name": row["sport_name"],
                    "target": row["target"],
                    "dx_callsign": row["dx_callsign"],
                    "power": row["power"],
                    "distance": row["distance"],
                    "date": row["date"],
                    "my_sig_info": row["my_sig_info"],
                })

    # Sort by value descending
    distance_records.sort(key=lambda x: x["value"], reverse=True)
    cool_factor_records.sort(key=lambda x: x["value"], reverse=True)

    return distance_records, cool_factor_records
