"""
Maidenhead Grid Square and Distance Calculations.

Reference implementations for:
1. Converting Maidenhead grid squares to lat/long
2. Calculating great-circle distance between two points
"""

import math
from typing import Tuple


def maidenhead_to_latlon(grid: str) -> Tuple[float, float]:
    """
    Convert a Maidenhead grid square to latitude/longitude (center of grid).

    Supports 2, 4, 6, or 8 character grid squares.

    The Maidenhead system:
    - First pair (field): A-R, represents 20° longitude and 10° latitude
    - Second pair (square): 0-9, represents 2° longitude and 1° latitude
    - Third pair (subsquare): A-X, represents 5' longitude and 2.5' latitude
    - Fourth pair (extended): 0-9, represents 0.5' longitude and 0.25' latitude

    Args:
        grid: Maidenhead grid square (e.g., "FN31", "FN31pr", "EM12ab34")

    Returns:
        Tuple of (latitude, longitude) in decimal degrees

    Raises:
        ValueError: If grid format is invalid
    """
    grid = grid.upper().strip()

    if len(grid) < 2 or len(grid) % 2 != 0 or len(grid) > 8:
        raise ValueError(f"Invalid grid length: {len(grid)}. Must be 2, 4, 6, or 8 characters.")

    # Validate characters
    if not ('A' <= grid[0] <= 'R' and 'A' <= grid[1] <= 'R'):
        raise ValueError(f"Invalid field characters: {grid[0:2]}. Must be A-R.")

    # Field (first pair): 20° lon, 10° lat
    lon = (ord(grid[0]) - ord('A')) * 20 - 180
    lat = (ord(grid[1]) - ord('A')) * 10 - 90

    # Default to center of field
    lon_offset = 10.0
    lat_offset = 5.0

    if len(grid) >= 4:
        if not (grid[2].isdigit() and grid[3].isdigit()):
            raise ValueError(f"Invalid square characters: {grid[2:4]}. Must be 0-9.")

        # Square (second pair): 2° lon, 1° lat
        lon += int(grid[2]) * 2
        lat += int(grid[3]) * 1

        lon_offset = 1.0
        lat_offset = 0.5

    if len(grid) >= 6:
        if not ('A' <= grid[4].upper() <= 'X' and 'A' <= grid[5].upper() <= 'X'):
            raise ValueError(f"Invalid subsquare characters: {grid[4:6]}. Must be A-X.")

        # Subsquare (third pair): 5' lon, 2.5' lat
        lon += (ord(grid[4].upper()) - ord('A')) * (5.0 / 60.0)
        lat += (ord(grid[5].upper()) - ord('A')) * (2.5 / 60.0)

        lon_offset = 2.5 / 60.0
        lat_offset = 1.25 / 60.0

    if len(grid) >= 8:
        if not (grid[6].isdigit() and grid[7].isdigit()):
            raise ValueError(f"Invalid extended characters: {grid[6:8]}. Must be 0-9.")

        # Extended (fourth pair): 0.5' lon, 0.25' lat
        lon += int(grid[6]) * (0.5 / 60.0)
        lat += int(grid[7]) * (0.25 / 60.0)

        lon_offset = 0.25 / 60.0
        lat_offset = 0.125 / 60.0

    # Return center of grid square
    return (lat + lat_offset, lon + lon_offset)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points using Haversine formula.

    Args:
        lat1, lon1: First point coordinates in decimal degrees
        lat2, lon2: Second point coordinates in decimal degrees

    Returns:
        Distance in kilometers
    """
    # Earth's radius in kilometers
    R = 6371.0

    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def grid_distance(grid1: str, grid2: str) -> float:
    """
    Calculate distance between two Maidenhead grid squares.

    Args:
        grid1: First grid square
        grid2: Second grid square

    Returns:
        Distance in kilometers
    """
    lat1, lon1 = maidenhead_to_latlon(grid1)
    lat2, lon2 = maidenhead_to_latlon(grid2)
    return haversine_distance(lat1, lon1, lat2, lon2)
