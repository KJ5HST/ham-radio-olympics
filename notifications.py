"""
Push Notification Service for Ham Radio Olympics.

Handles Web Push notifications for:
- Medal changes (won/lost medals)
- New QSO confirmations
- Record broken (personal or world)
- Match reminders (7 days before start)
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from pywebpush import webpush, WebPushException

from database import get_db

logger = logging.getLogger(__name__)

# VAPID keys for Web Push
# Generate with: vapid --gen (from py_vapid package)
# Or use: from pywebpush import generate_vapid_keypair; print(generate_vapid_keypair())
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "")
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", "")
VAPID_CLAIMS = {
    "sub": os.getenv("VAPID_SUBJECT", "mailto:admin@hamradio-olympics.com")
}


@dataclass
class NotificationPayload:
    """Push notification payload structure."""
    title: str
    body: str
    icon: str = "/static/icon-192.png"
    badge: str = "/static/icon-96.png"
    tag: str = "hro-notification"
    url: str = "/"
    renotify: bool = False
    actions: Optional[List[Dict[str, str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "title": self.title,
            "body": self.body,
            "icon": self.icon,
            "badge": self.badge,
            "tag": self.tag,
            "url": self.url,
            "renotify": self.renotify,
        }
        if self.actions:
            data["actions"] = self.actions
        return data


def get_vapid_public_key() -> str:
    """Get the VAPID public key for client subscription."""
    return VAPID_PUBLIC_KEY


def is_push_configured() -> bool:
    """Check if push notifications are properly configured."""
    return bool(VAPID_PRIVATE_KEY and VAPID_PUBLIC_KEY)


def save_subscription(
    callsign: str,
    endpoint: str,
    p256dh_key: str,
    auth_key: str,
    user_agent: Optional[str] = None
) -> bool:
    """Save a push subscription for a user."""
    try:
        with get_db() as conn:
            now = datetime.utcnow().isoformat()

            # Insert or update subscription
            conn.execute("""
                INSERT INTO push_subscriptions
                    (callsign, endpoint, p256dh_key, auth_key, user_agent, created_at, last_used_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(endpoint) DO UPDATE SET
                    callsign = excluded.callsign,
                    p256dh_key = excluded.p256dh_key,
                    auth_key = excluded.auth_key,
                    user_agent = excluded.user_agent,
                    last_used_at = excluded.last_used_at
            """, (callsign, endpoint, p256dh_key, auth_key, user_agent, now, now))

            # Initialize notification preferences if not exists
            conn.execute("""
                INSERT OR IGNORE INTO notification_preferences
                    (callsign, medal_changes, new_confirmations, record_broken, match_reminders, updated_at)
                VALUES (?, 1, 1, 1, 1, ?)
            """, (callsign, now))

            return True
    except Exception as e:
        logger.error(f"Failed to save subscription for {callsign}: {e}")
        return False


def remove_subscription(endpoint: str) -> bool:
    """Remove a push subscription by endpoint."""
    try:
        with get_db() as conn:
            conn.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))
            return True
    except Exception as e:
        logger.error(f"Failed to remove subscription: {e}")
        return False


def remove_all_subscriptions(callsign: str) -> bool:
    """Remove all push subscriptions for a user."""
    try:
        with get_db() as conn:
            conn.execute("DELETE FROM push_subscriptions WHERE callsign = ?", (callsign,))
            return True
    except Exception as e:
        logger.error(f"Failed to remove subscriptions for {callsign}: {e}")
        return False


def get_subscriptions(callsign: str) -> List[Dict[str, str]]:
    """Get all push subscriptions for a user."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT endpoint, p256dh_key, auth_key
            FROM push_subscriptions
            WHERE callsign = ?
        """, (callsign,)).fetchall()

        return [
            {
                "endpoint": row["endpoint"],
                "keys": {
                    "p256dh": row["p256dh_key"],
                    "auth": row["auth_key"]
                }
            }
            for row in rows
        ]


def get_notification_preferences(callsign: str) -> Dict[str, bool]:
    """Get notification preferences for a user."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT medal_changes, new_confirmations, record_broken, match_reminders
            FROM notification_preferences
            WHERE callsign = ?
        """, (callsign,)).fetchone()

        if row:
            return {
                "medal_changes": bool(row["medal_changes"]),
                "new_confirmations": bool(row["new_confirmations"]),
                "record_broken": bool(row["record_broken"]),
                "match_reminders": bool(row["match_reminders"])
            }

        # Default all enabled
        return {
            "medal_changes": True,
            "new_confirmations": True,
            "record_broken": True,
            "match_reminders": True
        }


def update_notification_preferences(callsign: str, preferences: Dict[str, bool]) -> bool:
    """Update notification preferences for a user."""
    try:
        with get_db() as conn:
            now = datetime.utcnow().isoformat()
            conn.execute("""
                INSERT INTO notification_preferences
                    (callsign, medal_changes, new_confirmations, record_broken, match_reminders, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(callsign) DO UPDATE SET
                    medal_changes = excluded.medal_changes,
                    new_confirmations = excluded.new_confirmations,
                    record_broken = excluded.record_broken,
                    match_reminders = excluded.match_reminders,
                    updated_at = excluded.updated_at
            """, (
                callsign,
                int(preferences.get("medal_changes", True)),
                int(preferences.get("new_confirmations", True)),
                int(preferences.get("record_broken", True)),
                int(preferences.get("match_reminders", True)),
                now
            ))
            return True
    except Exception as e:
        logger.error(f"Failed to update preferences for {callsign}: {e}")
        return False


def _send_push(subscription: Dict[str, Any], payload: NotificationPayload) -> bool:
    """Send a push notification to a single subscription."""
    if not is_push_configured():
        logger.warning("Push notifications not configured (missing VAPID keys)")
        return False

    try:
        webpush(
            subscription_info=subscription,
            data=json.dumps(payload.to_dict()),
            vapid_private_key=VAPID_PRIVATE_KEY,
            vapid_claims=VAPID_CLAIMS
        )
        return True
    except WebPushException as e:
        logger.error(f"Push failed: {e}")
        # If subscription is invalid (410 Gone or 404), remove it
        if e.response and e.response.status_code in (404, 410):
            remove_subscription(subscription.get("endpoint", ""))
        return False
    except Exception as e:
        logger.error(f"Push error: {e}")
        return False


def send_notification(callsign: str, payload: NotificationPayload) -> int:
    """Send notification to all of a user's subscriptions. Returns count of successful sends."""
    subscriptions = get_subscriptions(callsign)
    success_count = 0

    for sub in subscriptions:
        if _send_push(sub, payload):
            success_count += 1
            # Update last_used_at
            try:
                with get_db() as conn:
                    conn.execute(
                        "UPDATE push_subscriptions SET last_used_at = ? WHERE endpoint = ?",
                        (datetime.utcnow().isoformat(), sub["endpoint"])
                    )
            except Exception:
                pass

    return success_count


def was_notification_sent(callsign: str, notification_type: str, reference_id: str) -> bool:
    """Check if a notification was already sent (to avoid duplicates)."""
    with get_db() as conn:
        row = conn.execute("""
            SELECT id FROM sent_notifications
            WHERE callsign = ? AND notification_type = ? AND reference_id = ?
        """, (callsign, notification_type, reference_id)).fetchone()
        return row is not None


def mark_notification_sent(callsign: str, notification_type: str, reference_id: str) -> None:
    """Record that a notification was sent."""
    try:
        with get_db() as conn:
            conn.execute("""
                INSERT INTO sent_notifications (callsign, notification_type, reference_id, sent_at)
                VALUES (?, ?, ?, ?)
            """, (callsign, notification_type, reference_id, datetime.utcnow().isoformat()))
    except Exception as e:
        logger.error(f"Failed to mark notification sent: {e}")


# ============================================================================
# Notification Type Helpers
# ============================================================================

def notify_medal_change(
    callsign: str,
    sport_name: str,
    match_target: str,
    old_medals: Dict[str, Optional[str]],
    new_medals: Dict[str, Optional[str]],
    total_points: int
) -> bool:
    """
    Send notification about medal changes.

    Args:
        callsign: User's callsign
        sport_name: Name of the sport
        match_target: Target value (e.g., "EU", "K-0001")
        old_medals: Dict with 'qso_race' and 'cool_factor' old medal values
        new_medals: Dict with 'qso_race' and 'cool_factor' new medal values
        total_points: New total points
    """
    prefs = get_notification_preferences(callsign)
    if not prefs.get("medal_changes"):
        return False

    # Determine what changed
    changes = []
    medal_emoji = {"gold": "🥇", "silver": "🥈", "bronze": "🥉"}

    for category in ["qso_race", "cool_factor"]:
        old = old_medals.get(category)
        new = new_medals.get(category)
        cat_name = "QSO Race" if category == "qso_race" else "Cool Factor"

        if new and not old:
            # Won a medal
            changes.append(f"Won {medal_emoji.get(new, '')} {new.title()} in {cat_name}")
        elif old and not new:
            # Lost a medal
            changes.append(f"Lost {cat_name} medal")
        elif old and new and old != new:
            # Medal changed
            if ["gold", "silver", "bronze"].index(new) < ["gold", "silver", "bronze"].index(old):
                changes.append(f"Upgraded to {medal_emoji.get(new, '')} {new.title()} in {cat_name}")
            else:
                changes.append(f"Medal changed to {medal_emoji.get(new, '')} {new.title()} in {cat_name}")

    if not changes:
        return False

    title = f"Medal Update - {match_target}"
    body = f"{sport_name}: {'; '.join(changes)}. Total: {total_points} pts"

    payload = NotificationPayload(
        title=title,
        body=body,
        tag=f"medal-{sport_name}-{match_target}",
        url=f"/dashboard",
        renotify=True,
        actions=[
            {"action": "view", "title": "View Standings"}
        ]
    )

    return send_notification(callsign, payload) > 0


def notify_new_confirmations(
    callsign: str,
    confirmation_count: int,
    sample_calls: List[str]
) -> bool:
    """
    Send notification about new QSO confirmations.

    Args:
        callsign: User's callsign
        confirmation_count: Number of new confirmations
        sample_calls: List of sample DX callsigns (up to 3)
    """
    prefs = get_notification_preferences(callsign)
    if not prefs.get("new_confirmations"):
        return False

    if confirmation_count == 0:
        return False

    # Create a unique reference to avoid duplicate notifications
    today = datetime.utcnow().strftime("%Y-%m-%d")
    reference = f"conf-{today}-{confirmation_count}"

    if was_notification_sent(callsign, "confirmation", reference):
        return False

    if confirmation_count == 1:
        title = "New QSO Confirmation"
        body = f"Your QSO with {sample_calls[0]} has been confirmed!"
    else:
        sample_text = ", ".join(sample_calls[:3])
        if confirmation_count > 3:
            sample_text += f" and {confirmation_count - 3} more"
        title = f"{confirmation_count} New Confirmations"
        body = f"QSOs confirmed: {sample_text}"

    payload = NotificationPayload(
        title=title,
        body=body,
        tag="confirmations",
        url="/dashboard",
        actions=[
            {"action": "view", "title": "View QSOs"}
        ]
    )

    result = send_notification(callsign, payload) > 0
    if result:
        mark_notification_sent(callsign, "confirmation", reference)
    return result


def notify_record_broken(
    callsign: str,
    record_type: str,
    value: float,
    is_world_record: bool,
    sport_name: Optional[str] = None,
    dx_callsign: Optional[str] = None
) -> bool:
    """
    Send notification about a broken record.

    Args:
        callsign: User's callsign
        record_type: Type of record (e.g., "distance", "cool_factor")
        value: The record value
        is_world_record: Whether this is a world record
        sport_name: Name of sport (for sport-specific records)
        dx_callsign: DX station callsign
    """
    prefs = get_notification_preferences(callsign)
    if not prefs.get("record_broken"):
        return False

    record_emoji = "🌍" if is_world_record else "🏆"
    record_label = "World Record" if is_world_record else "Personal Best"

    if record_type == "distance":
        value_str = f"{value:,.0f} km"
        type_name = "Distance"
    elif record_type == "cool_factor":
        value_str = f"{value:.1f} km/W"
        type_name = "Cool Factor"
    else:
        value_str = f"{value:.1f}"
        type_name = record_type.replace("_", " ").title()

    title = f"{record_emoji} New {record_label}!"
    body = f"{type_name}: {value_str}"
    if dx_callsign:
        body += f" (QSO with {dx_callsign})"
    if sport_name:
        body += f" in {sport_name}"

    payload = NotificationPayload(
        title=title,
        body=body,
        tag=f"record-{record_type}",
        url="/records",
        renotify=True,
        actions=[
            {"action": "view", "title": "View Records"}
        ]
    )

    return send_notification(callsign, payload) > 0


def notify_match_reminder(
    callsign: str,
    sport_name: str,
    match_target: str,
    start_date: str,
    days_until: int
) -> bool:
    """
    Send notification about upcoming match.

    Args:
        callsign: User's callsign
        sport_name: Name of the sport
        match_target: Target value
        start_date: Match start date (YYYY-MM-DD)
        days_until: Days until match starts
    """
    prefs = get_notification_preferences(callsign)
    if not prefs.get("match_reminders"):
        return False

    # Create reference to avoid duplicate reminders
    reference = f"reminder-{sport_name}-{match_target}-{start_date}"

    if was_notification_sent(callsign, "match_reminder", reference):
        return False

    if days_until == 0:
        title = f"Match Starting Today!"
        time_text = "starts today"
    elif days_until == 1:
        title = f"Match Starting Tomorrow"
        time_text = "starts tomorrow"
    else:
        title = f"Match in {days_until} Days"
        time_text = f"starts in {days_until} days"

    body = f"{sport_name}: {match_target} {time_text}"

    payload = NotificationPayload(
        title=title,
        body=body,
        tag=f"reminder-{sport_name}",
        url="/olympiad/sports",
        actions=[
            {"action": "view", "title": "View Details"}
        ]
    )

    result = send_notification(callsign, payload) > 0
    if result:
        mark_notification_sent(callsign, "match_reminder", reference)
    return result


def send_match_reminders() -> Dict[str, int]:
    """
    Check for upcoming matches and send reminders.
    Should be called daily by a scheduled task.

    Returns dict with counts of reminders sent.
    """
    results = {"checked": 0, "sent": 0, "skipped": 0}

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

        # Get all users with push subscriptions
        subscribers = conn.execute("""
            SELECT DISTINCT callsign
            FROM push_subscriptions
        """).fetchall()

        for match in matches:
            match_date = datetime.fromisoformat(match["start_date"]).date()
            days_until = (match_date - today).days
            results["checked"] += 1

            for sub in subscribers:
                callsign = sub["callsign"]

                if notify_match_reminder(
                    callsign=callsign,
                    sport_name=match["sport_name"],
                    match_target=match["target_value"],
                    start_date=match["start_date"],
                    days_until=days_until
                ):
                    results["sent"] += 1
                else:
                    results["skipped"] += 1

    return results


def cleanup_old_notifications(days: int = 30) -> int:
    """Remove old sent_notifications records. Returns count deleted."""
    try:
        with get_db() as conn:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            cursor = conn.execute(
                "DELETE FROM sent_notifications WHERE sent_at < ?",
                (cutoff,)
            )
            return cursor.rowcount
    except Exception as e:
        logger.error(f"Failed to cleanup notifications: {e}")
        return 0


def cleanup_stale_subscriptions(days: int = 90) -> int:
    """Remove subscriptions not used in X days. Returns count deleted."""
    try:
        with get_db() as conn:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            cursor = conn.execute(
                "DELETE FROM push_subscriptions WHERE last_used_at < ? OR last_used_at IS NULL",
                (cutoff,)
            )
            return cursor.rowcount
    except Exception as e:
        logger.error(f"Failed to cleanup subscriptions: {e}")
        return 0
