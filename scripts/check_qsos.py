#!/usr/bin/env python3
"""Check QSO data for a competitor."""
import sqlite3
import sys

callsign = sys.argv[1] if len(sys.argv) > 1 else "KB5UTY"

conn = sqlite3.connect("/data/ham_olympics.db")
cursor = conn.execute("""
    SELECT qso_datetime_utc, dx_callsign, my_grid, my_sig_info, dx_sig_info, is_confirmed
    FROM qsos
    WHERE competitor_callsign = ?
    ORDER BY qso_datetime_utc DESC
    LIMIT 20
""", (callsign,))

print(f"{callsign} Recent QSOs:")
print("-" * 95)
print(f"{'Date':<22} {'DX Call':<12} {'MY_GRID':<10} {'MY_SIG_INFO':<16} {'DX_SIG_INFO':<16} {'Conf'}")
print("-" * 95)

for row in cursor.fetchall():
    dt = row[0] or "-"
    dx = row[1] or "-"
    my_grid = row[2] or "-"
    my_sig = row[3] or "-"
    dx_sig = row[4] or "-"
    conf = "Yes" if row[5] else "No"
    print(f"{dt:<22} {dx:<12} {my_grid:<10} {my_sig:<16} {dx_sig:<16} {conf}")

# Check their sync settings
print("\n" + "=" * 50)
cursor = conn.execute("""
    SELECT qrz_api_key_encrypted IS NOT NULL as has_qrz,
           lotw_username_encrypted IS NOT NULL as has_lotw
    FROM competitors WHERE callsign = ?
""", (callsign,))
row = cursor.fetchone()
if row:
    print(f"QRZ API Key: {'Yes' if row[0] else 'No'}")
    print(f"LoTW Credentials: {'Yes' if row[1] else 'No'}")
