#!/usr/bin/env python3
"""
Send match reminder push notifications.

This script should be run daily via cron or scheduled task.
It checks for matches starting within 7 days and sends reminders
to users who have push notifications enabled.

Usage:
    python scripts/send_match_reminders.py

Can also be triggered via admin endpoint: POST /admin/notifications/send-reminders
"""

import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from notifications import send_match_reminders, cleanup_old_notifications, cleanup_stale_subscriptions
from notifications import discord_notify_match_reminder, is_discord_configured


def send_discord_match_reminders() -> dict:
    """Send Discord notifications for upcoming matches."""
    from database import get_db
    from datetime import datetime, timedelta

    results = {"checked": 0, "sent": 0, "skipped": 0}

    if not is_discord_configured():
        return results

    with get_db() as conn:
        # Get active olympiad
        olympiad = conn.execute(
            "SELECT id FROM olympiads WHERE is_active = 1"
        ).fetchone()

        if not olympiad:
            return results

        # Find matches starting within next 7 days
        today = datetime.utcnow().date()
        future_date = today + timedelta(days=7)

        matches = conn.execute("""
            SELECT m.id, m.target_value, m.start_date, s.name as sport_name
            FROM matches m
            JOIN sports s ON m.sport_id = s.id
            WHERE s.olympiad_id = ?
              AND m.start_date BETWEEN ? AND ?
        """, (olympiad["id"], today.isoformat(), future_date.isoformat())).fetchall()

        # Check if we already sent a Discord reminder for each match today
        for match in matches:
            results["checked"] += 1
            match_date = datetime.fromisoformat(match["start_date"]).date()
            days_until = (match_date - today).days

            # Only send Discord reminders for 7, 1, and 0 days out
            if days_until not in [0, 1, 7]:
                results["skipped"] += 1
                continue

            # Create a reference to avoid duplicates (daily)
            reference = f"discord-{match['id']}-{today.isoformat()}"
            existing = conn.execute(
                "SELECT id FROM sent_notifications WHERE callsign = 'DISCORD' AND reference_id = ?",
                (reference,)
            ).fetchone()

            if existing:
                results["skipped"] += 1
                continue

            if discord_notify_match_reminder(
                sport_name=match["sport_name"],
                match_target=match["target_value"],
                days_until=days_until,
                start_date=match["start_date"]
            ):
                # Mark as sent
                conn.execute(
                    "INSERT INTO sent_notifications (callsign, notification_type, reference_id, sent_at) VALUES (?, ?, ?, ?)",
                    ("DISCORD", "match_reminder", reference, datetime.utcnow().isoformat())
                )
                results["sent"] += 1
            else:
                results["skipped"] += 1

    return results


def main():
    print("Ham Radio Olympics - Match Reminder Service")
    print("=" * 50)

    # Send match reminders (push notifications)
    print("\nSending push notification reminders...")
    results = send_match_reminders()
    print(f"  Matches checked: {results['checked']}")
    print(f"  Reminders sent: {results['sent']}")
    print(f"  Reminders skipped (already sent or disabled): {results['skipped']}")

    # Send Discord reminders
    print("\nSending Discord reminders...")
    discord_results = send_discord_match_reminders()
    print(f"  Matches checked: {discord_results['checked']}")
    print(f"  Discord reminders sent: {discord_results['sent']}")
    print(f"  Discord reminders skipped: {discord_results['skipped']}")

    # Cleanup old notification records
    print("\nCleaning up old notification records...")
    deleted = cleanup_old_notifications(days=30)
    print(f"  Deleted {deleted} old notification records")

    # Cleanup stale subscriptions
    print("\nCleaning up stale push subscriptions...")
    deleted = cleanup_stale_subscriptions(days=90)
    print(f"  Deleted {deleted} stale subscriptions")

    print("\n" + "=" * 50)
    print("Match reminder service complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
