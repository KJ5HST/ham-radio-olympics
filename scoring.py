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


def normalize_mode(mode: str) -> str:
    """
    Normalize QSO mode for comparison.

    Groups similar modes together:
    - USB, LSB -> SSB
    - Various digital modes stay as-is
    """
    if not mode:
        return ""
    mode = mode.upper().strip()
    # Normalize sideband modes to SSB
    if mode in ("USB", "LSB"):
        return "SSB"
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
    match_allowed_modes: str = None,
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

    Returns:
        List of MatchingQSO objects
    """
    matching = []

    target_type = sport_config["target_type"]
    work_enabled = sport_config["work_enabled"]
    activate_enabled = sport_config["activate_enabled"]
    separate_pools = sport_config["separate_pools"]
    # Match-level modes override sport-level if specified
    allowed_modes = match_allowed_modes if match_allowed_modes else sport_config.get("allowed_modes")

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

            # Check if QSO mode is allowed (applies to both work and activate)
            if not is_mode_allowed(qso.get("mode"), allowed_modes):
                continue

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

        sport_config = {
            "target_type": match_data["target_type"],
            "work_enabled": bool(match_data["work_enabled"]),
            "activate_enabled": bool(match_data["activate_enabled"]),
            "separate_pools": bool(match_data["separate_pools"]),
            "allowed_modes": match_data.get("sport_allowed_modes"),
        }

        # Get matching QSOs (only from competitors who opted into this sport)
        # Match-level allowed_modes overrides sport-level if set
        matching = get_matching_qsos(
            match_id,
            sport_config,
            match_data["target_value"],
            start_date,
            end_date,
            sport_id=match_data["sport_id"],
            match_allowed_modes=match_data.get("allowed_modes"),
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


def update_records(qso_id: int, callsign: str, sport_id: Optional[int] = None, match_id: Optional[int] = None):
    """
    Check and update records based on a QSO.

    Args:
        qso_id: QSO ID
        callsign: Competitor callsign
        sport_id: Sport ID (None for global records)
        match_id: Match ID where the QSO qualified
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
                qso_id, callsign, sport_id, match_id, achieved_at, higher_is_better=True
            )

        # Check highest cool factor
        if qso["cool_factor"]:
            _update_record_if_better(
                conn, "highest_cool_factor", qso["cool_factor"],
                qso_id, callsign, sport_id, match_id, achieved_at, higher_is_better=True
            )

        # Check lowest power (only for confirmed QSOs with positive power)
        if qso["tx_power_w"] and qso["tx_power_w"] > 0 and qso["is_confirmed"]:
            _update_record_if_better(
                conn, "lowest_power", qso["tx_power_w"],
                qso_id, callsign, sport_id, match_id, achieved_at, higher_is_better=False
            )


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
            INSERT INTO records (sport_id, match_id, callsign, record_type, value, qso_id, achieved_at)
            VALUES (?, ?, NULL, ?, ?, ?, ?)
        """, (sport_id, match_id, record_type, value, qso_id, achieved_at))
    else:
        is_better = (value > world_record["value"]) if higher_is_better else (value < world_record["value"])
        if is_better:
            conn.execute("""
                UPDATE records SET value = ?, qso_id = ?, match_id = ?, achieved_at = ?
                WHERE id = ?
            """, (value, qso_id, match_id, achieved_at, world_record["id"]))

    # Check personal best (per callsign)
    cursor = conn.execute("""
        SELECT id, value FROM records
        WHERE record_type = ? AND callsign = ? AND sport_id IS ?
    """, (record_type, callsign, sport_id))
    pb_record = cursor.fetchone()

    if pb_record is None:
        conn.execute("""
            INSERT INTO records (sport_id, match_id, callsign, record_type, value, qso_id, achieved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (sport_id, match_id, callsign, record_type, value, qso_id, achieved_at))
    else:
        is_better = (value > pb_record["value"]) if higher_is_better else (value < pb_record["value"])
        if is_better:
            conn.execute("""
                UPDATE records SET value = ?, qso_id = ?, match_id = ?, achieved_at = ?
                WHERE id = ?
            """, (value, qso_id, match_id, achieved_at, pb_record["id"]))


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
    """
    with get_db() as conn:
        # Get all sports
        cursor = conn.execute("SELECT id FROM sports")
        sport_ids = [row["id"] for row in cursor.fetchall()]

    for sport_id in sport_ids:
        # Compute sport-level standings
        compute_team_standings(sport_id)

        # Compute match-level standings
        with get_db() as conn:
            cursor = conn.execute(
                "SELECT id FROM matches WHERE sport_id = ?",
                (sport_id,)
            )
            match_ids = [row["id"] for row in cursor.fetchall()]

        for match_id in match_ids:
            compute_team_standings(sport_id, match_id)


def recompute_all_records():
    """
    Recompute all records from scratch using only QSOs that qualified for matches.

    This clears existing records and rebuilds them based on QSOs that actually
    matched a target during an active match period.
    """
    with get_db() as conn:
        # Clear all existing records
        conn.execute("DELETE FROM records")

        # Get all matches with their sport config
        cursor = conn.execute("""
            SELECT m.id as match_id, m.start_date, m.end_date, m.target_value,
                   m.allowed_modes as match_allowed_modes,
                   s.id as sport_id, s.target_type, s.work_enabled, s.activate_enabled,
                   s.separate_pools, s.allowed_modes as sport_allowed_modes, o.qualifying_qsos
            FROM matches m
            JOIN sports s ON m.sport_id = s.id
            JOIN olympiads o ON s.olympiad_id = o.id
        """)
        matches = [dict(row) for row in cursor.fetchall()]

    # Process each match to find qualifying QSOs
    seen_qsos = set()  # Track QSOs we've already processed for records

    for match_data in matches:
        start_date = datetime.fromisoformat(match_data["start_date"])
        end_date = datetime.fromisoformat(match_data["end_date"])

        sport_config = {
            "target_type": match_data["target_type"],
            "work_enabled": bool(match_data["work_enabled"]),
            "activate_enabled": bool(match_data["activate_enabled"]),
            "separate_pools": bool(match_data["separate_pools"]),
            "allowed_modes": match_data.get("sport_allowed_modes"),
        }

        # Get matching QSOs for this match
        matching = get_matching_qsos(
            match_data["match_id"],
            sport_config,
            match_data["target_value"],
            start_date,
            end_date,
            sport_id=match_data["sport_id"],
            match_allowed_modes=match_data.get("match_allowed_modes"),
        )

        # Update records for each matching QSO (only once per QSO)
        for qso in matching:
            if qso.qso_id not in seen_qsos:
                seen_qsos.add(qso.qso_id)
                update_records(qso.qso_id, qso.callsign, match_id=match_data["match_id"])
