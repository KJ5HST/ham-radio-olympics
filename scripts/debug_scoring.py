#!/usr/bin/env python3
"""Debug scoring for a specific match."""
import sys
sys.path.insert(0, "/app")
import sqlite3
from scoring import matches_target, is_mode_allowed, validate_qso_for_mode, get_matching_qsos
from datetime import datetime

conn = sqlite3.connect("/data/ham_olympics.db")
conn.row_factory = sqlite3.Row

# Get KB5UTY first QSO from US-3013 activation
cursor = conn.execute("""
    SELECT * FROM qsos
    WHERE competitor_callsign = 'KB5UTY' AND my_sig_info = 'US-3013' AND is_confirmed = 1
    LIMIT 1
""")
qso = dict(cursor.fetchone())

print("QSO data:")
print(f"  mode: {qso.get('mode')}")
print(f"  my_sig_info: {qso.get('my_sig_info')}")
print(f"  is_confirmed: {qso.get('is_confirmed')}")
print(f"  distance_km: {qso.get('distance_km')}")
print(f"  tx_power_w: {qso.get('tx_power_w')}")
print(f"  cool_factor: {qso.get('cool_factor')}")

# Check mode allowed
print(f"\nis_mode_allowed('{qso.get('mode')}', 'SSB'): {is_mode_allowed(qso.get('mode'), 'SSB')}")

# Check validate_qso_for_mode
valid, error = validate_qso_for_mode(qso, "activate")
print(f"validate_qso_for_mode(activate): valid={valid}, error={error}")

# Check matches_target
matches = matches_target(qso, "park", "US-3013", "activate")
print(f"matches_target(park, US-3013, activate): {matches}")

# Check if KB5UTY is in sport 5
cursor = conn.execute("SELECT * FROM sport_entries WHERE callsign = 'KB5UTY' AND sport_id = 5")
entry = cursor.fetchone()
print(f"\nKB5UTY in sport 5: {entry is not None}")

# Now test get_matching_qsos
print("\n" + "="*60)
print("Testing get_matching_qsos for match 141 (US-3013):")

# Get sport config
cursor = conn.execute("""
    SELECT s.target_type, s.work_enabled, s.activate_enabled, s.separate_pools, s.allowed_modes
    FROM sports s JOIN matches m ON m.sport_id = s.id WHERE m.id = 141
""")
row = cursor.fetchone()
sport_config = {
    "target_type": row[0],
    "work_enabled": row[1],
    "activate_enabled": row[2],
    "separate_pools": row[3],
    "allowed_modes": row[4]
}
print(f"Sport config: {sport_config}")

# Get match details
cursor = conn.execute("SELECT target_value, start_date, end_date FROM matches WHERE id = 141")
match = cursor.fetchone()
print(f"Match: target={match[0]} start={match[1]} end={match[2]}")

qsos = get_matching_qsos(
    match_id=141,
    sport_config=sport_config,
    target_value=match[0],
    start_date=datetime.fromisoformat(match[1]),
    end_date=datetime.fromisoformat(match[2]),
    sport_id=5
)

print(f"\nFound {len(qsos)} matching QSOs:")
for q in qsos[:10]:
    print(f"  {q.callsign} {q.role} {q.qso_datetime} dist={q.distance_km} pwr={q.tx_power}")
