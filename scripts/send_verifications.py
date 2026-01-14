#!/usr/bin/env python3
"""Send verification emails to all unverified users."""
import asyncio
import sqlite3
import os
import sys

# Set up path
sys.path.insert(0, "/app")
os.chdir("/app")

from email_service import create_email_verification_token, send_email_verification

BASE_URL = "https://kd5dx.fly.dev"

async def send_all_verifications():
    conn = sqlite3.connect("/data/ham_olympics.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT callsign, email FROM competitors WHERE email IS NOT NULL AND email_verified = 0")
    unverified = cursor.fetchall()

    print(f"Sending verification emails to {len(unverified)} users...")

    for row in unverified:
        callsign = row[0]
        email = row[1]
        try:
            token = create_email_verification_token(callsign)
            verification_url = f"{BASE_URL}/verify-email/{token}"
            success = await send_email_verification(callsign, email, verification_url)
            if success:
                print(f"  Sent to {callsign} ({email})")
            else:
                print(f"  FAILED: {callsign} ({email})")
        except Exception as e:
            print(f"  ERROR {callsign}: {e}")
        # Rate limit: wait 1 second between emails
        await asyncio.sleep(1)

    print("Done!")

if __name__ == "__main__":
    asyncio.run(send_all_verifications())
