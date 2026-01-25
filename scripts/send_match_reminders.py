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


def main():
    print("Ham Radio Olympics - Match Reminder Service")
    print("=" * 50)

    # Send match reminders
    print("\nSending match reminders...")
    results = send_match_reminders()
    print(f"  Matches checked: {results['checked']}")
    print(f"  Reminders sent: {results['sent']}")
    print(f"  Reminders skipped (already sent or disabled): {results['skipped']}")

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
