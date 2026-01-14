#!/usr/bin/env python3
"""Send a test weekly digest email."""
import asyncio
import sys
sys.path.insert(0, "/app")

from email_service import send_email, _render_weekly_digest_template, _get_email_footer_text

async def send_test_digest():
    callsign = "KJ5HST"
    email = "kj5hst@deppe.com"

    # Sample matches for testing
    matches = [
        {"sport_name": "DX Challenge", "target_value": "EU", "start_date": "2026-01-15", "end_date": "2026-01-31"},
        {"sport_name": "POTA Championship", "target_value": "K-0001", "start_date": "2026-01-16", "end_date": "2026-01-22"},
        {"sport_name": "Grid Chase", "target_value": "FN31", "start_date": "2026-01-17", "end_date": "2026-01-24"},
    ]

    html_body = _render_weekly_digest_template(callsign, matches)

    plain_body = f"""
Upcoming Matches This Week

Hello {callsign},

Here are the upcoming matches in the Ham Radio Olympics:

- DX Challenge: EU (2026-01-15 - 2026-01-31)
- POTA Championship: K-0001 (2026-01-16 - 2026-01-22)
- Grid Chase: FN31 (2026-01-17 - 2026-01-24)

Get your rig ready and good luck!

73,
Ham Radio Olympics
{_get_email_footer_text()}
"""

    success = await send_email(
        to=email,
        subject="Upcoming Matches This Week - Ham Radio Olympics",
        body=plain_body.strip(),
        html_body=html_body
    )

    if success:
        print(f"Sent test weekly digest to {email}")
    else:
        print("Failed to send digest")

if __name__ == "__main__":
    asyncio.run(send_test_digest())
