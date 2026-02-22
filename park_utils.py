"""
Shared POTA park name utilities.

Provides DB-only lookups (no API calls) for park names.
Parks are cached in the pota_parks table during sync via validate_park_ids().
"""

import re
from typing import Optional

from database import get_db

# POTA park reference pattern: 1-3 letters, dash, 3+ digits
POTA_REFERENCE_PATTERN = re.compile(r'^[A-Z]{1,3}-\d{3,}$')


def is_pota_reference(value: str) -> bool:
    """Check if a string looks like a POTA park reference (e.g., K-0001, VE-0123)."""
    if not value:
        return False
    return bool(POTA_REFERENCE_PATTERN.match(value.upper().strip()))


def get_park_name_cached(reference: str) -> Optional[str]:
    """Look up park name from pota_parks table (DB cache only, no API calls)."""
    if not reference:
        return None
    reference = reference.upper().strip()
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT name FROM pota_parks WHERE reference = ?",
            (reference,)
        )
        row = cursor.fetchone()
        if row:
            return row["name"]
    return None


def format_park_display(park_id: str) -> str:
    """Format a park ID with its name if cached.

    Returns "Park Name (K-0001)" if name is cached, otherwise just "K-0001".
    """
    if not park_id:
        return park_id or ""
    park_id = park_id.strip()
    if not is_pota_reference(park_id):
        return park_id
    name = get_park_name_cached(park_id.upper())
    if name:
        return f"{name} ({park_id})"
    return park_id
