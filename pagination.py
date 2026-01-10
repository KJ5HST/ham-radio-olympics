"""
Pagination utilities for Ham Radio Olympics.
"""

import math
from typing import Dict, Any


class Paginator:
    """
    A reusable paginator class for handling pagination logic.
    """

    def __init__(self, page: int = 1, per_page: int = 50):
        """
        Initialize paginator.

        Args:
            page: Current page number (1-indexed)
            per_page: Number of items per page
        """
        self.page = max(1, page)  # Ensure page is at least 1
        self.per_page = max(1, per_page)  # Ensure per_page is at least 1

    def get_offset(self) -> int:
        """
        Calculate the offset for SQL LIMIT/OFFSET queries.

        Returns:
            The offset value
        """
        return (self.page - 1) * self.per_page

    def get_page_info(self, total: int) -> Dict[str, Any]:
        """
        Get pagination info for templates.

        Args:
            total: Total number of items

        Returns:
            Dictionary with pagination details
        """
        total_pages = math.ceil(total / self.per_page) if total > 0 else 0

        return {
            "current_page": self.page,
            "per_page": self.per_page,
            "total_items": total,
            "total_pages": total_pages,
            "has_prev": self.page > 1,
            "has_next": self.page < total_pages,
            "prev_page": self.page - 1 if self.page > 1 else None,
            "next_page": self.page + 1 if self.page < total_pages else None,
        }
