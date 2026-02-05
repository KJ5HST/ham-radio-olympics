#!/usr/bin/env python3
"""
Scheduled POTA spot checker.

Run every 30 minutes via cron or fly.io scheduler:
    fly machine run . --schedule "*/30 * * * *" -- python scripts/check_pota_spots.py

Or via cron:
    */30 * * * * cd /app && python scripts/check_pota_spots.py
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from notifications import check_pota_spots_and_notify

    print("Checking POTA spots...")
    results = await check_pota_spots_and_notify()

    print(f"Results:")
    print(f"  Spots checked: {results['spots_checked']}")
    print(f"  Sports with spots: {results['sports_with_spots']}")
    print(f"  Push notifications sent: {results['notifications_sent']}")
    print(f"  Discord summary sent: {results['discord_sent']}")
    print(f"  Errors: {results['errors']}")

    return results


if __name__ == "__main__":
    results = asyncio.run(main())
    sys.exit(0 if results["errors"] == 0 else 1)
