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
                    (callsign, medal_changes, new_confirmations, record_broken, match_reminders, pota_spots, updated_at)
                VALUES (?, 1, 1, 1, 1, 1, ?)
            """, (callsign, now))

            return True
    except Exception as e:
        logger.error(f"Failed to save subscription for {callsign}: {e}")
        return False


def remove_subscription(endpoint: str, notify_user: bool = False) -> bool:
    """
    Remove a push subscription by endpoint.

    Args:
        endpoint: The push subscription endpoint URL
        notify_user: If True, send email to user explaining how to re-enable

    Returns:
        True if subscription was removed
    """
    try:
        with get_db() as conn:
            # Get user info before deleting if we need to notify
            user_info = None
            if notify_user:
                cursor = conn.execute("""
                    SELECT ps.callsign, c.email, c.email_verified, c.email_notifications_enabled
                    FROM push_subscriptions ps
                    JOIN competitors c ON ps.callsign = c.callsign
                    WHERE ps.endpoint = ?
                """, (endpoint,))
                user_info = cursor.fetchone()

            conn.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))

            # Send notification email if requested and user has valid email
            if user_info and user_info["email"] and user_info["email_verified"] and user_info["email_notifications_enabled"]:
                # Schedule the email to be sent (we're in sync context, can't await)
                _pending_push_disabled_notifications.append({
                    "callsign": user_info["callsign"],
                    "email": user_info["email"]
                })

            return True
    except Exception as e:
        logger.error(f"Failed to remove subscription: {e}")
        return False


# Pending notifications to send after sync operations complete
_pending_push_disabled_notifications: List[Dict[str, str]] = []


async def send_pending_push_disabled_emails() -> int:
    """
    Send any pending push disabled notification emails.
    Call this after operations that might remove subscriptions.

    Returns:
        Number of emails sent
    """
    global _pending_push_disabled_notifications

    if not _pending_push_disabled_notifications:
        return 0

    from email_service import send_push_disabled_email

    sent = 0
    pending = _pending_push_disabled_notifications.copy()
    _pending_push_disabled_notifications.clear()

    for notification in pending:
        try:
            success = await send_push_disabled_email(
                callsign=notification["callsign"],
                email=notification["email"]
            )
            if success:
                sent += 1
                logger.info(f"Sent push disabled email to {notification['callsign']}")
        except Exception as e:
            logger.error(f"Failed to send push disabled email to {notification['callsign']}: {e}")

    return sent


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
            SELECT medal_changes, new_confirmations, record_broken, match_reminders, pota_spots
            FROM notification_preferences
            WHERE callsign = ?
        """, (callsign,)).fetchone()

        if row:
            return {
                "medal_changes": bool(row["medal_changes"]),
                "new_confirmations": bool(row["new_confirmations"]),
                "record_broken": bool(row["record_broken"]),
                "match_reminders": bool(row["match_reminders"]),
                "pota_spots": bool(row["pota_spots"]) if row["pota_spots"] is not None else True
            }

        # Default all enabled
        return {
            "medal_changes": True,
            "new_confirmations": True,
            "record_broken": True,
            "match_reminders": True,
            "pota_spots": True
        }


def update_notification_preferences(callsign: str, preferences: Dict[str, bool]) -> bool:
    """Update notification preferences for a user."""
    try:
        with get_db() as conn:
            now = datetime.utcnow().isoformat()
            conn.execute("""
                INSERT INTO notification_preferences
                    (callsign, medal_changes, new_confirmations, record_broken, match_reminders, pota_spots, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(callsign) DO UPDATE SET
                    medal_changes = excluded.medal_changes,
                    new_confirmations = excluded.new_confirmations,
                    record_broken = excluded.record_broken,
                    match_reminders = excluded.match_reminders,
                    pota_spots = excluded.pota_spots,
                    updated_at = excluded.updated_at
            """, (
                callsign,
                int(preferences.get("medal_changes", True)),
                int(preferences.get("new_confirmations", True)),
                int(preferences.get("record_broken", True)),
                int(preferences.get("match_reminders", True)),
                int(preferences.get("pota_spots", True)),
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
        error_str = str(e)
        logger.error(f"Push failed: {e}")

        # Remove invalid subscriptions:
        # - 400 Bad Request: Invalid/stale subscription (common with WNS/Windows)
        # - 404 Not Found: Subscription doesn't exist
        # - 410 Gone: Subscription expired/unsubscribed
        should_remove = False
        status_code = None

        # Try to get status code from response object
        if e.response and hasattr(e.response, 'status_code'):
            status_code = e.response.status_code
            if status_code in (400, 404, 410):
                should_remove = True

        # Fallback: parse status code from error message string
        # e.g., "Push failed: 400 Bad Request"
        if not should_remove:
            import re
            match = re.search(r'\b(400|404|410)\b', error_str)
            if match:
                status_code = int(match.group(1))
                should_remove = True

        if should_remove:
            endpoint = subscription.get("endpoint", "")
            logger.info(f"Removing invalid subscription (HTTP {status_code}): {endpoint[:50]}...")
            # Remove and notify user via email so they know how to re-enable
            remove_subscription(endpoint, notify_user=True)

        return False
    except Exception as e:
        logger.error(f"Push error: {e}")
        return False


# Track push errors to avoid spamming admin
_push_error_count = 0
_push_error_notified = False


async def notify_admin_push_errors(error_count: int, sample_errors: List[str]) -> None:
    """Notify admin about accumulated push notification errors."""
    global _push_error_notified
    if _push_error_notified:
        return

    try:
        from email_service import send_admin_error_email
        await send_admin_error_email(
            error_type="Push Notification Failures",
            error_details=f"{error_count} push notifications failed. This may indicate stale subscriptions or VAPID key issues.",
            context={
                "error_count": error_count,
                "sample_errors": "; ".join(sample_errors[:3]) if sample_errors else "No details"
            }
        )
        _push_error_notified = True
    except Exception as e:
        logger.error(f"Failed to notify admin of push errors: {e}")


def send_notification(callsign: str, payload: NotificationPayload) -> int:
    """Send notification to all of a user's subscriptions. Returns count of successful sends."""
    global _push_error_count
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
        else:
            _push_error_count += 1

    return success_count


def get_push_error_count() -> int:
    """Get the current push error count."""
    return _push_error_count


def reset_push_error_tracking() -> None:
    """Reset push error tracking (call after notifying admin)."""
    global _push_error_count, _push_error_notified
    _push_error_count = 0
    _push_error_notified = False


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
    days_until: int,
    sport_id: Optional[int] = None
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

    # Use sport page if sport_id is provided, otherwise dashboard
    url = f"/olympiad/sport/{sport_id}" if sport_id else "/dashboard"

    payload = NotificationPayload(
        title=title,
        body=body,
        tag=f"reminder-{sport_name}",
        url=url,
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
            SELECT m.id, m.target_value, m.start_date, s.id as sport_id, s.name as sport_name
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
                    days_until=days_until,
                    sport_id=match["sport_id"]
                ):
                    results["sent"] += 1
                else:
                    results["skipped"] += 1

    return results


def notify_pota_spot(
    callsign: str,
    park_reference: str,
    activator_callsign: str,
    frequency: str,
    mode: str,
    sport_name: str,
    sport_id: Optional[int] = None
) -> bool:
    """
    Send notification about a POTA spot for an active match.

    Args:
        callsign: User's callsign to notify
        park_reference: POTA park reference (e.g., "K-0001")
        activator_callsign: Callsign of the activator
        frequency: Frequency of the spot
        mode: Mode (SSB, CW, FT8, etc.)
        sport_name: Name of the sport/match
    """
    prefs = get_notification_preferences(callsign)
    if not prefs.get("pota_spots"):
        return False

    # Create reference to avoid duplicate notifications for same activator at same park
    # Daily dedupe - one notification per activator/park per day
    today = datetime.utcnow().strftime("%Y%m%d")
    reference = f"spot-{park_reference}-{activator_callsign}-{today}"

    if was_notification_sent(callsign, "pota_spot", reference):
        return False

    title = f"POTA Spot: {park_reference}"
    body = f"{activator_callsign} on {frequency} {mode} - {sport_name}"

    # Use sport page if sport_id is provided, otherwise dashboard
    url = f"/olympiad/sport/{sport_id}" if sport_id else "/dashboard"

    payload = NotificationPayload(
        title=title,
        body=body,
        tag=f"spot-{park_reference}",
        url=url,
        renotify=True,
        actions=[
            {"action": "view", "title": "View Match"}
        ]
    )

    result = send_notification(callsign, payload) > 0
    if result:
        mark_notification_sent(callsign, "pota_spot", reference)
    return result


async def check_pota_spots_and_notify() -> Dict[str, int]:
    """
    Check POTA spots for active matches and notify users.

    Should be called on a schedule (every 30 minutes).

    - Discord: Sends a single batched summary (e.g., "POTA Championship: 5 spots")
    - Push: Notifies individual users about spots (daily dedupe per activator/park)

    Returns dict with counts of spots found and notifications sent.
    """
    import httpx
    from config import config

    results = {"spots_checked": 0, "sports_with_spots": 0, "notifications_sent": 0, "discord_sent": 0, "errors": 0}

    try:
        # Fetch current POTA spots
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.pota.app/spot/activator",
                timeout=10.0
            )
            if response.status_code != 200:
                logger.warning(f"Failed to fetch POTA spots: HTTP {response.status_code}")
                return results

            all_spots = response.json()
            if not all_spots:
                return results

            results["spots_checked"] = len(all_spots)

        # Build a dict of park -> spots for quick lookup
        spots_by_park = {}
        for spot in all_spots:
            park_ref = spot.get("reference")
            if park_ref:
                if park_ref not in spots_by_park:
                    spots_by_park[park_ref] = []
                spots_by_park[park_ref].append(spot)

        with get_db() as conn:
            # Get active olympiad
            olympiad = conn.execute(
                "SELECT id FROM olympiads WHERE is_active = 1"
            ).fetchone()

            if not olympiad:
                return results

            # Get active matches with park targets (POTA)
            now = datetime.utcnow().isoformat()
            matches = conn.execute("""
                SELECT m.id, m.target_value, s.id as sport_id, s.name as sport_name, s.allowed_modes
                FROM matches m
                JOIN sports s ON m.sport_id = s.id
                WHERE s.olympiad_id = ?
                  AND s.target_type = 'park'
                  AND m.start_date <= ?
                  AND m.end_date >= ?
            """, (olympiad["id"], now, now)).fetchall()

            # Import mode filter
            from scoring import is_mode_allowed

            # Collect spot counts per sport for Discord summary
            sport_spot_counts: Dict[str, Dict[str, Any]] = {}

            for match in matches:
                park_ref = match["target_value"]
                all_park_spots = spots_by_park.get(park_ref, [])

                if not all_park_spots:
                    continue

                sport_name = match["sport_name"]
                sport_id = match["sport_id"]
                allowed_modes = match["allowed_modes"]

                # Filter spots by allowed modes for this sport
                if allowed_modes:
                    spots = [s for s in all_park_spots if is_mode_allowed(s.get("mode", ""), allowed_modes)]
                else:
                    spots = all_park_spots

                if not spots:
                    continue

                # Count spots per sport (not per match)
                if sport_name not in sport_spot_counts:
                    sport_spot_counts[sport_name] = {"count": 0, "sport_id": sport_id}
                sport_spot_counts[sport_name]["count"] += len(spots)

                # Push notifications: notify each subscriber (daily dedupe)
                subscribers = conn.execute("""
                    SELECT DISTINCT se.callsign
                    FROM sport_entries se
                    JOIN push_subscriptions ps ON se.callsign = ps.callsign
                    WHERE se.sport_id = ?
                """, (sport_id,)).fetchall()

                for sub in subscribers:
                    user_callsign = sub["callsign"]

                    for spot in spots:
                        activator = spot.get("activator", "Unknown")
                        # Don't notify user about their own activation
                        if activator.upper() == user_callsign.upper():
                            continue

                        frequency = spot.get("frequency", "?")
                        mode = spot.get("mode", "?")

                        try:
                            if notify_pota_spot(
                                callsign=user_callsign,
                                park_reference=park_ref,
                                activator_callsign=activator,
                                frequency=frequency,
                                mode=mode,
                                sport_name=sport_name,
                                sport_id=sport_id
                            ):
                                results["notifications_sent"] += 1
                        except Exception as e:
                            logger.error(f"Failed to send POTA spot notification to {user_callsign}: {e}")
                            results["errors"] += 1

            # Discord: send batched summary if there are any spots
            if sport_spot_counts:
                results["sports_with_spots"] = len(sport_spot_counts)

                # Deduplicate Discord summary (30-minute window)
                time_bucket = datetime.utcnow().strftime("%Y%m%d%H") + str(datetime.utcnow().minute // 30)
                discord_ref = f"discord-pota-summary-{time_bucket}"

                if not was_notification_sent("_discord_", "pota_summary", discord_ref):
                    try:
                        app_url = config.APP_BASE_URL.rstrip("/")
                        if discord_notify_pota_summary(sport_spot_counts, app_url):
                            results["discord_sent"] = 1
                            mark_notification_sent("_discord_", "pota_summary", discord_ref)
                    except Exception as e:
                        logger.error(f"Failed to send Discord POTA summary: {e}")

    except httpx.RequestError as e:
        logger.error(f"Failed to fetch POTA spots: {e}")
        results["errors"] += 1
    except Exception as e:
        logger.error(f"Error checking POTA spots: {e}")
        results["errors"] += 1

    if results["notifications_sent"] > 0 or results["discord_sent"] > 0:
        logger.info(f"POTA spot notifications: {results['notifications_sent']} push, {results['discord_sent']} Discord for {results['sports_with_spots']} sports")

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


# ============================================================================
# Discord Webhook Notifications
# ============================================================================

def get_discord_webhook_url() -> Optional[str]:
    """Get Discord webhook URL from settings."""
    from database import get_setting
    return get_setting("discord_webhook_url")


def is_discord_configured() -> bool:
    """Check if Discord webhook is configured."""
    url = get_discord_webhook_url()
    return bool(url and url.strip())


def send_discord_notification(embed: Dict[str, Any]) -> bool:
    """
    Send a notification to Discord webhook.

    Args:
        embed: Discord embed object with title, description, color, fields, etc.

    Returns:
        True if sent successfully, False otherwise.
    """
    webhook_url = get_discord_webhook_url()
    if not webhook_url:
        return False

    try:
        import httpx

        payload = {"embeds": [embed]}
        response = httpx.post(
            webhook_url,
            json=payload,
            timeout=10.0
        )
        if response.status_code in (200, 204):
            return True
        else:
            logger.error(f"Discord webhook failed with status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Discord webhook error: {e}")
        return False


def discord_notify_record(
    callsign: str,
    record_type: str,
    value: float,
    is_world_record: bool,
    sport_name: Optional[str] = None,
    dx_callsign: Optional[str] = None
) -> bool:
    """
    Send Discord notification for a new record.

    Args:
        callsign: The competitor's callsign
        record_type: Type of record (distance, cool_factor, lowest_power)
        value: The record value
        is_world_record: Whether this is a world record (vs personal best)
        sport_name: Name of the sport (optional)
        dx_callsign: DX station worked (optional)
    """
    if not is_discord_configured():
        return False

    # Only notify on world records for Discord (to avoid spam)
    if not is_world_record:
        return False

    # Deduplicate: only notify once per record type/value combination
    # Use value in reference so a NEW world record still triggers
    sport_key = sport_name or "global"
    reference = f"discord-record-{record_type}-{sport_key}-{value}"
    if was_notification_sent("_discord_", "record", reference):
        return False

    # Format the value based on type
    if record_type == "distance":
        value_str = f"{value:,.0f} km"
        type_name = "Distance"
        emoji = "📏"
    elif record_type == "cool_factor":
        value_str = f"{value:.1f} km/W"
        type_name = "Cool Factor"
        emoji = "❄️"
    elif record_type == "lowest_power":
        value_str = f"{value:.1f} W"
        type_name = "Lowest Power DX"
        emoji = "🔋"
    else:
        value_str = f"{value:.1f}"
        type_name = record_type.replace("_", " ").title()
        emoji = "🏆"

    title = f"🥇 New World Record! {emoji}"
    description = f"**{callsign}** set a new {type_name.lower()} record!"

    fields = [
        {"name": type_name, "value": value_str, "inline": True}
    ]
    if dx_callsign:
        fields.append({"name": "QSO With", "value": dx_callsign, "inline": True})
    if sport_name:
        fields.append({"name": "Sport", "value": sport_name, "inline": True})

    embed = {
        "title": title,
        "description": description,
        "color": 0xFFD700,  # Gold
        "fields": fields,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    result = send_discord_notification(embed)
    if result:
        mark_notification_sent("_discord_", "record", reference)
    return result


def discord_notify_medal(
    callsign: str,
    sport_name: str,
    match_target: str,
    medal_type: str,
    competition: str
) -> bool:
    """
    Send Discord notification for a medal award.

    Args:
        callsign: The competitor's callsign
        sport_name: Name of the sport
        match_target: Target value (e.g., "EU", "K-0001")
        medal_type: "gold", "silver", or "bronze"
        competition: "QSO Race" or "Cool Factor"
    """
    if not is_discord_configured():
        return False

    # Deduplicate: only notify once per callsign/sport/target/medal/competition
    reference = f"discord-medal-{callsign}-{sport_name}-{match_target}-{medal_type}-{competition}"
    if was_notification_sent("_discord_", "medal", reference):
        return False

    medal_colors = {
        "gold": 0xFFD700,
        "silver": 0xC0C0C0,
        "bronze": 0xCD7F32
    }
    medal_emojis = {
        "gold": "🥇",
        "silver": "🥈",
        "bronze": "🥉"
    }

    emoji = medal_emojis.get(medal_type, "🏅")
    color = medal_colors.get(medal_type, 0x808080)

    title = f"{emoji} {medal_type.title()} Medal Awarded!"
    description = f"**{callsign}** earned {medal_type} in {competition}"

    embed = {
        "title": title,
        "description": description,
        "color": color,
        "fields": [
            {"name": "Sport", "value": sport_name, "inline": True},
            {"name": "Target", "value": match_target, "inline": True},
            {"name": "Competition", "value": competition, "inline": True}
        ],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    result = send_discord_notification(embed)
    if result:
        mark_notification_sent("_discord_", "medal", reference)
    return result


def discord_notify_signup(callsign: str) -> bool:
    """
    Send Discord notification for a new competitor signup.

    Args:
        callsign: The new competitor's callsign
    """
    if not is_discord_configured():
        return False

    embed = {
        "title": "👋 New Competitor Joined!",
        "description": f"Welcome **{callsign}** to Ham Radio Olympics!",
        "color": 0x00FF00,  # Green
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    return send_discord_notification(embed)


def discord_notify_match_reminder(
    sport_name: str,
    match_target: str,
    days_until: int,
    start_date: str
) -> bool:
    """
    Send Discord notification for an upcoming match.

    Args:
        sport_name: Name of the sport
        match_target: Target value (e.g., "EU", "K-0001")
        days_until: Number of days until match starts
        start_date: Match start date (YYYY-MM-DD)
    """
    if not is_discord_configured():
        return False

    if days_until == 0:
        title = "🚀 Match Starting Today!"
        time_text = "starts **today**"
        color = 0xFF0000  # Red - urgent
    elif days_until == 1:
        title = "⏰ Match Starting Tomorrow!"
        time_text = "starts **tomorrow**"
        color = 0xFFA500  # Orange
    else:
        title = f"📅 Match in {days_until} Days"
        time_text = f"starts in **{days_until} days**"
        color = 0x0099FF  # Blue

    embed = {
        "title": title,
        "description": f"**{sport_name}** targeting **{match_target}** {time_text}",
        "color": color,
        "fields": [
            {"name": "Start Date", "value": start_date, "inline": True}
        ],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    return send_discord_notification(embed)


def discord_notify_pota_summary(
    sport_spot_counts: Dict[str, Dict[str, Any]],
    app_url: str = "https://kd5dx.fly.dev"
) -> bool:
    """
    Send a batched Discord notification summarizing POTA activity across sports.

    Args:
        sport_spot_counts: Dict mapping sport_name -> {"count": int, "sport_id": int}
        app_url: Base URL for the app

    Example message:
        📡 POTA Activity
        • POTA Championship: 5 active spots
        • Park Pursuit: 2 active spots
        [View on Ham Radio Olympics]
    """
    if not is_discord_configured():
        return False

    if not sport_spot_counts:
        return False

    # Build description with sport names and counts
    lines = []
    for sport_name, info in sorted(sport_spot_counts.items()):
        count = info["count"]
        sport_id = info["sport_id"]
        spot_word = "spot" if count == 1 else "spots"
        lines.append(f"• [{sport_name}]({app_url}/olympiad/sport/{sport_id}): **{count}** active {spot_word}")

    total = sum(info["count"] for info in sport_spot_counts.values())

    embed = {
        "title": f"📡 POTA Activity ({total} spots)",
        "description": "\n".join(lines),
        "color": 0x00AA00,  # Green
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    return send_discord_notification(embed)


def test_discord_webhook() -> Dict[str, Any]:
    """
    Send a test message to the configured Discord webhook.

    Returns:
        Dict with success status and message
    """
    if not is_discord_configured():
        return {"success": False, "message": "Discord webhook URL not configured"}

    embed = {
        "title": "🔔 Test Notification",
        "description": "This is a test message from Ham Radio Olympics!",
        "color": 0x7289DA,  # Discord blurple
        "fields": [
            {"name": "Status", "value": "Webhook is working correctly", "inline": False}
        ],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    success = send_discord_notification(embed)
    if success:
        return {"success": True, "message": "Test notification sent successfully!"}
    else:
        return {"success": False, "message": "Failed to send test notification. Check the webhook URL."}
