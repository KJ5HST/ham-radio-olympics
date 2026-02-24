"""
Fix duplicate QSOs where one record has sig_info and the other doesn't.

These duplicates are caused by LoTW (no park data) and QRZ (has park data)
creating separate records instead of merging.

Strategy: For each NULL-sig record, find the best matching non-NULL record
and merge them. This correctly handles two-fers (one NULL LoTW record should
merge with only one of the QRZ park records).

Usage:
  python3 scripts/fix_sig_info_dupes.py          # dry run
  python3 scripts/fix_sig_info_dupes.py --fix     # actually merge
"""
import os
import sys
import sqlite3

db_path = os.getenv("DATABASE_PATH", "ham_olympics.db")
dry_run = "--fix" not in sys.argv

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

# Find records with NULL sig_info that have a matching record with non-NULL sig_info
# The NULL record is always the "source" to merge FROM (it's the LoTW record missing park data)
# We merge it INTO the first matching non-NULL record
null_records = conn.execute("""
    SELECT a.id, a.competitor_callsign, a.dx_callsign, a.qso_datetime_utc, a.mode
    FROM qsos a
    WHERE a.dx_sig_info IS NULL AND a.my_sig_info IS NULL
    AND EXISTS (
        SELECT 1 FROM qsos b
        WHERE b.competitor_callsign = a.competitor_callsign
        AND b.dx_callsign = a.dx_callsign
        AND b.id != a.id
        AND (b.dx_sig_info IS NOT NULL OR b.my_sig_info IS NOT NULL)
        AND (b.mode = a.mode OR b.mode IS NULL OR a.mode IS NULL)
        AND ABS(CAST((julianday(a.qso_datetime_utc) - julianday(b.qso_datetime_utc)) * 86400 AS INTEGER)) <= 120
    )
    ORDER BY a.competitor_callsign, a.qso_datetime_utc
""").fetchall()

print(f"Found {len(null_records)} NULL-sig records with matching non-NULL duplicates")
if dry_run:
    print("DRY RUN - pass --fix to actually merge\n")

merged = 0
deleted_ids = set()

for nr in null_records:
    null_id = nr["id"]
    if null_id in deleted_ids:
        continue

    # Find the best matching record with park data (prefer one with qrz_logid)
    match = conn.execute("""
        SELECT id, dx_sig_info, my_sig_info, qrz_logid
        FROM qsos
        WHERE competitor_callsign = ? AND dx_callsign = ?
        AND id != ?
        AND (dx_sig_info IS NOT NULL OR my_sig_info IS NOT NULL)
        AND (mode = ? OR mode IS NULL OR ? IS NULL)
        AND ABS(CAST((julianday(qso_datetime_utc) - julianday(?)) * 86400 AS INTEGER)) <= 120
        ORDER BY qrz_logid IS NOT NULL DESC, id ASC
        LIMIT 1
    """, (nr["competitor_callsign"], nr["dx_callsign"], null_id,
          nr["mode"], nr["mode"], nr["qso_datetime_utc"])).fetchone()

    if not match or match["id"] in deleted_ids:
        continue

    target_id = match["id"]
    print(f"  {nr['competitor_callsign']} -> {nr['dx_callsign']} @ {nr['qso_datetime_utc']}")
    print(f"    NULL record id={null_id} -> merge INTO id={target_id} (park={match['dx_sig_info'] or match['my_sig_info']})")

    if not dry_run:
        # Merge NULL record's fields into the target (COALESCE keeps existing non-NULL values)
        null_rec = dict(conn.execute("SELECT * FROM qsos WHERE id = ?", (null_id,)).fetchone())

        conn.execute("""
            UPDATE qsos SET
                is_confirmed = CASE WHEN ? = 1 OR is_confirmed = 1 THEN 1 ELSE 0 END,
                confirmed_at = COALESCE(confirmed_at, ?),
                band = COALESCE(band, ?),
                mode = COALESCE(mode, ?),
                tx_power_w = COALESCE(tx_power_w, ?),
                my_dxcc = COALESCE(my_dxcc, ?),
                my_grid = CASE WHEN LENGTH(COALESCE(?, '')) > LENGTH(COALESCE(my_grid, '')) THEN ? ELSE my_grid END,
                my_sig_info = COALESCE(my_sig_info, ?),
                dx_dxcc = COALESCE(dx_dxcc, ?),
                dx_grid = CASE WHEN LENGTH(COALESCE(?, '')) > LENGTH(COALESCE(dx_grid, '')) THEN ? ELSE dx_grid END,
                dx_sig_info = COALESCE(dx_sig_info, ?),
                distance_km = COALESCE(distance_km, ?),
                cool_factor = COALESCE(cool_factor, ?),
                qrz_logid = COALESCE(qrz_logid, ?)
            WHERE id = ?
        """, (
            null_rec["is_confirmed"],
            null_rec.get("confirmed_at"),
            null_rec.get("band"),
            null_rec.get("mode"),
            null_rec.get("tx_power_w"),
            null_rec.get("my_dxcc"),
            null_rec.get("my_grid"), null_rec.get("my_grid"),
            null_rec.get("my_sig_info"),
            null_rec.get("dx_dxcc"),
            null_rec.get("dx_grid"), null_rec.get("dx_grid"),
            null_rec.get("dx_sig_info"),
            null_rec.get("distance_km"),
            null_rec.get("cool_factor"),
            null_rec.get("qrz_logid"),
            target_id,
        ))
        conn.execute("DELETE FROM qsos WHERE id = ?", (null_id,))
        deleted_ids.add(null_id)
        merged += 1
        print(f"    -> MERGED, deleted {null_id}")

if not dry_run:
    conn.commit()
    print(f"\nMerged and deleted {merged} duplicate NULL-sig records")
else:
    print(f"\nWould merge {len(null_records)} records (run with --fix)")

conn.close()
