"""
Audit logging for Ham Radio Olympics.

Provides functions to log and retrieve audit events for security and compliance.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from database import get_db


def log_action(
    actor_callsign: str,
    action: str,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    details: Optional[str] = None,
    ip_address: Optional[str] = None
) -> None:
    """
    Log an audit action.

    Args:
        actor_callsign: The callsign of the user performing the action
        action: The type of action (e.g., 'login', 'logout', 'password_change')
        target_type: The type of target (e.g., 'competitor', 'sport')
        target_id: The ID of the target
        details: Additional details about the action
        ip_address: The IP address of the actor
    """
    timestamp = datetime.utcnow().isoformat()

    with get_db() as conn:
        conn.execute("""
            INSERT INTO audit_log (timestamp, actor_callsign, action, target_type, target_id, details, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, actor_callsign, action, target_type, target_id, details, ip_address))


def get_audit_logs(
    limit: int = 100,
    offset: int = 0,
    action: Optional[str] = None,
    actor_callsign: Optional[str] = None,
    target_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve audit log entries.

    Args:
        limit: Maximum number of entries to return
        offset: Number of entries to skip
        action: Filter by action type
        actor_callsign: Filter by actor callsign
        target_type: Filter by target type

    Returns:
        List of audit log entries as dictionaries
    """
    query = """
        SELECT a.*, c.first_name as actor_first_name
        FROM audit_log a
        LEFT JOIN competitors c ON a.actor_callsign = c.callsign
        WHERE 1=1
    """
    params = []

    if action:
        query += " AND a.action = ?"
        params.append(action)

    if actor_callsign:
        query += " AND a.actor_callsign = ?"
        params.append(actor_callsign)

    if target_type:
        query += " AND a.target_type = ?"
        params.append(target_type)

    query += " ORDER BY a.timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_db() as conn:
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_audit_log_count(
    action: Optional[str] = None,
    actor_callsign: Optional[str] = None,
    target_type: Optional[str] = None
) -> int:
    """
    Get the total count of audit log entries matching filters.

    Args:
        action: Filter by action type
        actor_callsign: Filter by actor callsign
        target_type: Filter by target type

    Returns:
        Total count of matching entries
    """
    query = "SELECT COUNT(*) as count FROM audit_log WHERE 1=1"
    params = []

    if action:
        query += " AND action = ?"
        params.append(action)

    if actor_callsign:
        query += " AND actor_callsign = ?"
        params.append(actor_callsign)

    if target_type:
        query += " AND target_type = ?"
        params.append(target_type)

    with get_db() as conn:
        cursor = conn.execute(query, params)
        return cursor.fetchone()["count"]
