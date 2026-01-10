"""
Callsign lookup service for fetching operator names and country info.
Uses HamQTH (free) for lookups with local caching.
"""

import httpx
import xml.etree.ElementTree as ET
from typing import Optional, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass

from database import get_db


HAMQTH_URL = "https://www.hamqth.com/xml.php"
CACHE_DURATION_DAYS = 30


@dataclass
class CallsignInfo:
    """Callsign lookup result."""
    callsign: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    country: Optional[str] = None
    dxcc: Optional[int] = None
    grid: Optional[str] = None


# Country to flag emoji mapping (ISO 3166-1 alpha-2 to regional indicator)
COUNTRY_FLAGS = {
    "United States": "🇺🇸",
    "USA": "🇺🇸",
    "Canada": "🇨🇦",
    "United Kingdom": "🇬🇧",
    "England": "🇬🇧",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
    "Germany": "🇩🇪",
    "France": "🇫🇷",
    "Italy": "🇮🇹",
    "Spain": "🇪🇸",
    "Japan": "🇯🇵",
    "Australia": "🇦🇺",
    "New Zealand": "🇳🇿",
    "Brazil": "🇧🇷",
    "Argentina": "🇦🇷",
    "Mexico": "🇲🇽",
    "Russia": "🇷🇺",
    "China": "🇨🇳",
    "India": "🇮🇳",
    "South Africa": "🇿🇦",
    "Netherlands": "🇳🇱",
    "Belgium": "🇧🇪",
    "Switzerland": "🇨🇭",
    "Austria": "🇦🇹",
    "Poland": "🇵🇱",
    "Czech Republic": "🇨🇿",
    "Sweden": "🇸🇪",
    "Norway": "🇳🇴",
    "Denmark": "🇩🇰",
    "Finland": "🇫🇮",
    "Ireland": "🇮🇪",
    "Portugal": "🇵🇹",
    "Greece": "🇬🇷",
    "Ukraine": "🇺🇦",
    "Croatia": "🇭🇷",
    "Slovenia": "🇸🇮",
    "Romania": "🇷🇴",
    "Hungary": "🇭🇺",
    "Bulgaria": "🇧🇬",
    "Serbia": "🇷🇸",
    "Slovakia": "🇸🇰",
    "Lithuania": "🇱🇹",
    "Latvia": "🇱🇻",
    "Estonia": "🇪🇪",
    "Israel": "🇮🇱",
    "Turkey": "🇹🇷",
    "South Korea": "🇰🇷",
    "Korea": "🇰🇷",
    "Taiwan": "🇹🇼",
    "Philippines": "🇵🇭",
    "Thailand": "🇹🇭",
    "Indonesia": "🇮🇩",
    "Malaysia": "🇲🇾",
    "Singapore": "🇸🇬",
    "Vietnam": "🇻🇳",
    "Chile": "🇨🇱",
    "Colombia": "🇨🇴",
    "Peru": "🇵🇪",
    "Venezuela": "🇻🇪",
    "Ecuador": "🇪🇨",
    "Uruguay": "🇺🇾",
    "Paraguay": "🇵🇾",
    "Bolivia": "🇧🇴",
    "Puerto Rico": "🇵🇷",
    "Cuba": "🇨🇺",
    "Jamaica": "🇯🇲",
    "Costa Rica": "🇨🇷",
    "Panama": "🇵🇦",
    "Guatemala": "🇬🇹",
    "Honduras": "🇭🇳",
    "El Salvador": "🇸🇻",
    "Nicaragua": "🇳🇮",
    "Dominican Republic": "🇩🇴",
    "Haiti": "🇭🇹",
    "Trinidad and Tobago": "🇹🇹",
    "Barbados": "🇧🇧",
    "Bahamas": "🇧🇸",
    "Bermuda": "🇧🇲",
    "Cayman Islands": "🇰🇾",
    "Aruba": "🇦🇼",
    "Curacao": "🇨🇼",
    "Egypt": "🇪🇬",
    "Morocco": "🇲🇦",
    "Algeria": "🇩🇿",
    "Tunisia": "🇹🇳",
    "Libya": "🇱🇾",
    "Nigeria": "🇳🇬",
    "Kenya": "🇰🇪",
    "Ghana": "🇬🇭",
    "Zimbabwe": "🇿🇼",
    "Zambia": "🇿🇲",
    "Botswana": "🇧🇼",
    "Namibia": "🇳🇦",
    "Mozambique": "🇲🇿",
    "Tanzania": "🇹🇿",
    "Uganda": "🇺🇬",
    "Ethiopia": "🇪🇹",
    "Saudi Arabia": "🇸🇦",
    "United Arab Emirates": "🇦🇪",
    "UAE": "🇦🇪",
    "Kuwait": "🇰🇼",
    "Qatar": "🇶🇦",
    "Bahrain": "🇧🇭",
    "Oman": "🇴🇲",
    "Jordan": "🇯🇴",
    "Lebanon": "🇱🇧",
    "Iraq": "🇮🇶",
    "Iran": "🇮🇷",
    "Pakistan": "🇵🇰",
    "Bangladesh": "🇧🇩",
    "Sri Lanka": "🇱🇰",
    "Nepal": "🇳🇵",
    "Mongolia": "🇲🇳",
    "Kazakhstan": "🇰🇿",
    "Uzbekistan": "🇺🇿",
    "Guam": "🇬🇺",
    "Hawaii": "🇺🇸",
    "Alaska": "🇺🇸",
    "Iceland": "🇮🇸",
    "Luxembourg": "🇱🇺",
    "Malta": "🇲🇹",
    "Cyprus": "🇨🇾",
    "Monaco": "🇲🇨",
    "Andorra": "🇦🇩",
    "San Marino": "🇸🇲",
    "Liechtenstein": "🇱🇮",
    "Vatican": "🇻🇦",
    "Fed. Rep. of Germany": "🇩🇪",
    # Additional territories and countries
    "Aland Islands": "🇦🇽",
    "Canary Islands": "🇮🇨",
    "Ceuta & Melilla": "🇪🇸",
    "Corsica": "🇫🇷",
    "Sardinia": "🇮🇹",
    "Madeira Islands": "🇵🇹",
    "Northern Ireland": "🇬🇧",
    "Mariana Islands": "🇲🇵",
    "Gabon": "🇬🇦",
    "Lesotho": "🇱🇸",
    "Liberia": "🇱🇷",
    "Malawi": "🇲🇼",
    "Mauritania": "🇲🇷",
    "Reunion": "🇷🇪",
    "Rwanda": "🇷🇼",
    "Sierra Leone": "🇸🇱",
    "Bosnia-Herzegovina": "🇧🇦",
    "North Macedonia": "🇲🇰",
    "Yugoslavia": "🇷🇸",
    "Azores": "🇵🇹",
    "Balearic Islands": "🇪🇸",
    "Sicily": "🇮🇹",
    "Guadeloupe": "🇬🇵",
    "Martinique": "🇲🇶",
    "French Guiana": "🇬🇫",
    "New Caledonia": "🇳🇨",
    "French Polynesia": "🇵🇫",
    "Mayotte": "🇾🇹",
    "St. Pierre & Miquelon": "🇵🇲",
    "Wallis & Futuna": "🇼🇫",
    "Antarctica": "🇦🇶",
    "Greenland": "🇬🇱",
    "Faroe Islands": "🇫🇴",
    "Svalbard": "🇸🇯",
    "Jan Mayen": "🇸🇯",
    "Gibraltar": "🇬🇮",
    "Isle of Man": "🇮🇲",
    "Jersey": "🇯🇪",
    "Guernsey": "🇬🇬",
    "Falkland Islands": "🇫🇰",
    "South Georgia": "🇬🇸",
    "British Virgin Islands": "🇻🇬",
    "Turks & Caicos": "🇹🇨",
    "Montserrat": "🇲🇸",
    "Anguilla": "🇦🇮",
    "St. Kitts & Nevis": "🇰🇳",
    "Antigua & Barbuda": "🇦🇬",
    "Dominica": "🇩🇲",
    "St. Lucia": "🇱🇨",
    "St. Vincent": "🇻🇨",
    "Grenada": "🇬🇩",
    "Belize": "🇧🇿",
    "Guyana": "🇬🇾",
    "Suriname": "🇸🇷",
    "U.S. Virgin Islands": "🇻🇮",
    "American Samoa": "🇦🇸",
    "Midway Island": "🇺🇲",
    "Wake Island": "🇺🇲",
    "Johnston Island": "🇺🇲",
    "Marshall Islands": "🇲🇭",
    "Palau": "🇵🇼",
    "Micronesia": "🇫🇲",
    "Federated States of Micronesia": "🇫🇲",
    "Fiji": "🇫🇯",
    "Tonga": "🇹🇴",
    "Samoa": "🇼🇸",
    "Western Samoa": "🇼🇸",
    "Kiribati": "🇰🇮",
    "Tuvalu": "🇹🇻",
    "Vanuatu": "🇻🇺",
    "Solomon Islands": "🇸🇧",
    "Papua New Guinea": "🇵🇬",
    "Nauru": "🇳🇷",
    "Norfolk Island": "🇳🇫",
    "Cook Islands": "🇨🇰",
    "Niue": "🇳🇺",
    "Tokelau": "🇹🇰",
    "Christmas Island": "🇨🇽",
    "Cocos (Keeling) Islands": "🇨🇨",
    "Heard Island": "🇭🇲",
    "Macquarie Island": "🇦🇺",
    "Brunei": "🇧🇳",
    "Cambodia": "🇰🇭",
    "Laos": "🇱🇦",
    "Myanmar": "🇲🇲",
    "Burma": "🇲🇲",
    "Maldives": "🇲🇻",
    "Bhutan": "🇧🇹",
    "Timor-Leste": "🇹🇱",
    "East Timor": "🇹🇱",
    "Hong Kong": "🇭🇰",
    "Macau": "🇲🇴",
    "DPR of Korea": "🇰🇵",
    "North Korea": "🇰🇵",
    "Ogasawara": "🇯🇵",
    "Minami Torishima": "🇯🇵",
    "Asiatic Russia": "🇷🇺",
    "European Russia": "🇷🇺",
    "Franz Josef Land": "🇷🇺",
    "Kaliningrad": "🇷🇺",
    "Kyrgyzstan": "🇰🇬",
    "Tajikistan": "🇹🇯",
    "Turkmenistan": "🇹🇲",
    "Azerbaijan": "🇦🇿",
    "Georgia": "🇬🇪",
    "Armenia": "🇦🇲",
    "Moldova": "🇲🇩",
    "Belarus": "🇧🇾",
    "Eritrea": "🇪🇷",
    "Somalia": "🇸🇴",
    "Djibouti": "🇩🇯",
    "Comoros": "🇰🇲",
    "Mauritius": "🇲🇺",
    "Seychelles": "🇸🇨",
    "Madagascar": "🇲🇬",
    "Cape Verde": "🇨🇻",
    "Sao Tome & Principe": "🇸🇹",
    "Equatorial Guinea": "🇬🇶",
    "Cameroon": "🇨🇲",
    "Central African Republic": "🇨🇫",
    "Chad": "🇹🇩",
    "Republic of the Congo": "🇨🇬",
    "Democratic Republic of the Congo": "🇨🇩",
    "Zaire": "🇨🇩",
    "Burundi": "🇧🇮",
    "Benin": "🇧🇯",
    "Burkina Faso": "🇧🇫",
    "Cote d'Ivoire": "🇨🇮",
    "Ivory Coast": "🇨🇮",
    "Gambia": "🇬🇲",
    "Guinea": "🇬🇳",
    "Guinea-Bissau": "🇬🇼",
    "Mali": "🇲🇱",
    "Niger": "🇳🇪",
    "Senegal": "🇸🇳",
    "Togo": "🇹🇬",
    "Angola": "🇦🇴",
    "Sudan": "🇸🇩",
    "South Sudan": "🇸🇸",
    "Swaziland": "🇸🇿",
    "Eswatini": "🇸🇿",
    "St. Helena": "🇸🇭",
    "Ascension Island": "🇦🇨",
    "Tristan da Cunha": "🇹🇦",
    "Western Sahara": "🇪🇭",
    "Svalbard & Jan Mayen": "🇸🇯",
    "Bouvet Island": "🇧🇻",
    "Peter I Island": "🇦🇶",
    "South Orkney Islands": "🇦🇶",
    "South Shetland Islands": "🇦🇶",
    "Kerguelen Islands": "🇹🇫",
    "Crozet Island": "🇹🇫",
    "Amsterdam & St. Paul Is.": "🇹🇫",
    "Chagos Islands": "🇮🇴",
    "Diego Garcia": "🇮🇴",
    "Rodrigues Island": "🇲🇺",
    "Agalega & St. Brandon": "🇲🇺",
    "Glorioso Islands": "🇹🇫",
    "Juan de Nova & Europa": "🇹🇫",
    "Tromelin Island": "🇹🇫",
    "Prince Edward & Marion Is.": "🇿🇦",
    # More territories
    "East Malaysia": "🇲🇾",
    "Crete": "🇬🇷",
    "Chatham Islands": "🇳🇿",
    "Margarita Island": "🇻🇪",
    "Netherlands Antilles": "🇳🇱",
    "Aves Island": "🇻🇪",
    "Bear Island": "🇳🇴",
    "Pitcairn Island": "🇵🇳",
    "St. Barthelemy": "🇧🇱",
    "Clipperton Island": "🇫🇷",
    "St. Martin": "🇲🇫",
    "Bonaire": "🇧🇶",
    "Trinidad & Tobago": "🇹🇹",
}


def get_country_flag(country: Optional[str]) -> str:
    """Get flag emoji for a country name."""
    if not country:
        return ""
    return COUNTRY_FLAGS.get(country, "🌍")


def get_cached_callsign(callsign: str) -> Optional[CallsignInfo]:
    """Get callsign info from cache if not expired."""
    with get_db() as conn:
        cursor = conn.execute(
            """SELECT callsign, first_name, last_name, country, dxcc, grid, cached_at
               FROM callsign_cache WHERE callsign = ?""",
            (callsign.upper(),)
        )
        row = cursor.fetchone()
        if row:
            cached_at = datetime.fromisoformat(row["cached_at"])
            if datetime.utcnow() - cached_at < timedelta(days=CACHE_DURATION_DAYS):
                return CallsignInfo(
                    callsign=row["callsign"],
                    first_name=row["first_name"],
                    last_name=row["last_name"],
                    country=row["country"],
                    dxcc=row["dxcc"],
                    grid=row["grid"],
                )
    return None


def cache_callsign(info: CallsignInfo):
    """Store callsign info in cache."""
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO callsign_cache
               (callsign, first_name, last_name, country, dxcc, grid, cached_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                info.callsign.upper(),
                info.first_name,
                info.last_name,
                info.country,
                info.dxcc,
                info.grid,
                datetime.utcnow().isoformat(),
            )
        )


async def lookup_callsign_hamqth(callsign: str) -> Optional[CallsignInfo]:
    """Look up callsign using HamQTH (free, no auth required for basic info)."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                HAMQTH_URL,
                params={"id": callsign, "prg": "HamRadioOlympics"},
                timeout=10.0,
            )

            if response.status_code != 200:
                return None

            # Parse XML response
            root = ET.fromstring(response.text)

            # Check for error
            error = root.find(".//error")
            if error is not None:
                return None

            # Extract info
            search = root.find(".//search")
            if search is None:
                return None

            # Get name - might be in "nick" or need to parse "adr_name"
            nick = search.findtext("nick", "")
            adr_name = search.findtext("adr_name", "")

            first_name = None
            last_name = None

            if nick:
                first_name = nick
            elif adr_name:
                parts = adr_name.split()
                if parts:
                    first_name = parts[0]
                    if len(parts) > 1:
                        last_name = " ".join(parts[1:])

            country = search.findtext("country", None)
            grid = search.findtext("grid", None)

            return CallsignInfo(
                callsign=callsign.upper(),
                first_name=first_name,
                last_name=last_name,
                country=country,
                dxcc=None,  # HamQTH doesn't provide DXCC
                grid=grid,
            )
    except Exception:
        return None


async def lookup_callsign(callsign: str, use_cache: bool = True) -> Optional[CallsignInfo]:
    """
    Look up callsign information.

    Args:
        callsign: The callsign to look up
        use_cache: Whether to use cached results

    Returns:
        CallsignInfo if found, None otherwise
    """
    callsign = callsign.upper().strip()

    # Check cache first
    if use_cache:
        cached = get_cached_callsign(callsign)
        if cached:
            return cached

    # Try HamQTH
    info = await lookup_callsign_hamqth(callsign)

    if info:
        cache_callsign(info)
        return info

    return None


def get_display_name(callsign: str, first_name: Optional[str] = None) -> str:
    """
    Format display name as "FirstName (CALLSIGN)" or just "CALLSIGN".

    Args:
        callsign: The callsign
        first_name: Optional first name

    Returns:
        Formatted display string
    """
    if first_name:
        return f"{first_name} ({callsign})"
    return callsign


def get_dx_hover_text(
    callsign: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    country: Optional[str] = None,
) -> str:
    """
    Generate hover text for DX callsign.

    Args:
        callsign: The callsign
        first_name: Optional first name
        last_name: Optional last name
        country: Optional country name

    Returns:
        Hover text string
    """
    parts = []

    name = " ".join(filter(None, [first_name, last_name]))
    if name:
        parts.append(name)

    if country:
        flag = get_country_flag(country)
        parts.append(f"{flag} {country}")

    return " - ".join(parts) if parts else callsign
