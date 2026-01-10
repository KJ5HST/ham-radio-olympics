"""
DXCC entity to continent mapping.
"""

import json
import os
from typing import Optional

# Load DXCC data from JSON file
_dxcc_data = None


def _load_dxcc_data() -> dict:
    """Load DXCC data from JSON file."""
    global _dxcc_data
    if _dxcc_data is None:
        # Look for the file in the parent directory or current directory
        for path in ["../dxcc_continents.json", "dxcc_continents.json", "app/dxcc_continents.json"]:
            if os.path.exists(path):
                with open(path, "r") as f:
                    _dxcc_data = json.load(f)
                break

        if _dxcc_data is None:
            # Fallback to embedded data
            _dxcc_data = _get_embedded_data()

        # Merge countries from embedded data if not present
        if "countries" not in _dxcc_data:
            _dxcc_data["countries"] = _get_embedded_data()["countries"]

    return _dxcc_data


def get_continent(dxcc_code: int) -> Optional[str]:
    """
    Get the continent code for a DXCC entity.

    Args:
        dxcc_code: DXCC entity number

    Returns:
        Continent code (AF, AN, AS, EU, NA, OC, SA) or None if not found
    """
    data = _load_dxcc_data()
    entities = data.get("entities", {})
    return entities.get(str(dxcc_code))


def get_continent_name(code: str) -> Optional[str]:
    """
    Get the full continent name from code.

    Args:
        code: Continent code (e.g., "EU")

    Returns:
        Full continent name (e.g., "Europe") or None
    """
    data = _load_dxcc_data()
    continents = data.get("continents", {})
    return continents.get(code)


def get_all_continents() -> list:
    """
    Get all continents as (code, name) tuples sorted by name.

    Returns:
        List of (code, name) tuples
    """
    data = _load_dxcc_data()
    continents = data.get("continents", {})
    return sorted([(code, name) for code, name in continents.items()], key=lambda x: x[1])


def get_country_name(dxcc_code: int) -> Optional[str]:
    """
    Get the country name for a DXCC entity.

    Args:
        dxcc_code: DXCC entity number

    Returns:
        Country name or None if not found
    """
    data = _load_dxcc_data()
    countries = data.get("countries", {})
    return countries.get(str(dxcc_code))


def get_all_countries() -> list:
    """
    Get all DXCC countries as (code, name) tuples sorted by name.

    Returns:
        List of (dxcc_code, name) tuples
    """
    data = _load_dxcc_data()
    countries = data.get("countries", {})
    return sorted([(code, name) for code, name in countries.items()], key=lambda x: x[1])


def _get_embedded_data() -> dict:
    """Embedded DXCC data as fallback."""
    return {
        "continents": {
            "AF": "Africa",
            "AN": "Antarctica",
            "AS": "Asia",
            "EU": "Europe",
            "NA": "North America",
            "OC": "Oceania",
            "SA": "South America"
        },
        "countries": {
            "1": "Canada",
            "6": "Mexico",
            "9": "East Malaysia",
            "12": "Aruba",
            "13": "Antarctica",
            "15": "Asiatic Russia",
            "20": "Crete",
            "21": "Aland Islands",
            "27": "American Samoa",
            "29": "Canary Islands",
            "32": "Ceuta & Melilla",
            "45": "England",
            "46": "Niue",
            "47": "Cook Islands",
            "48": "Tokelau",
            "50": "Guam",
            "52": "Chatham Islands",
            "54": "Andorra",
            "62": "Bermuda",
            "63": "Curacao",
            "64": "Bahamas",
            "66": "Jamaica",
            "69": "Cayman Islands",
            "70": "Cuba",
            "71": "Margarita Island",
            "72": "Dominican Republic",
            "74": "Martinique",
            "78": "Puerto Rico",
            "79": "Guadeloupe",
            "80": "Haiti",
            "82": "Netherlands Antilles",
            "84": "Turks & Caicos",
            "86": "British Virgin Islands",
            "88": "St. Kitts & Nevis",
            "89": "U.S. Virgin Islands",
            "90": "Aves Island",
            "91": "Trinidad & Tobago",
            "94": "Antigua & Barbuda",
            "95": "Dominica",
            "97": "Aruba",
            "98": "St. Lucia",
            "100": "Argentina",
            "104": "Bolivia",
            "106": "Bear Island",
            "108": "Brazil",
            "110": "Nauru",
            "112": "Chile",
            "116": "Colombia",
            "118": "Jan Mayen",
            "120": "Ecuador",
            "122": "Svalbard",
            "129": "Falkland Islands",
            "130": "Maldives",
            "132": "Paraguay",
            "136": "Peru",
            "137": "Ogasawara",
            "144": "Uruguay",
            "146": "Monaco",
            "148": "Venezuela",
            "149": "Azores",
            "150": "Australia",
            "157": "Fiji",
            "160": "Hawaii",
            "162": "Kiribati",
            "163": "Indonesia",
            "165": "Mauritius",
            "166": "New Caledonia",
            "170": "New Zealand",
            "175": "Pitcairn Island",
            "177": "Papua New Guinea",
            "189": "Marshall Islands",
            "190": "Micronesia",
            "202": "Greenland",
            "203": "Luxembourg",
            "206": "Austria",
            "207": "Benin",
            "209": "Belgium",
            "212": "Bulgaria",
            "213": "Antigua & Barbuda",
            "214": "Corsica",
            "215": "Bangladesh",
            "221": "Denmark",
            "222": "Faroe Islands",
            "223": "Finland",
            "224": "France",
            "225": "Sardinia",
            "227": "Germany",
            "230": "Germany",
            "233": "Greece",
            "236": "Hungary",
            "237": "Montserrat",
            "239": "Iceland",
            "242": "Balearic Islands",
            "245": "Ireland",
            "246": "Isle of Man",
            "248": "Italy",
            "249": "St. Vincent",
            "251": "Vatican",
            "254": "Liechtenstein",
            "256": "Madeira Islands",
            "257": "Latvia",
            "263": "Netherlands",
            "265": "Northern Ireland",
            "266": "Norway",
            "269": "Poland",
            "272": "Portugal",
            "275": "Romania",
            "278": "San Marino",
            "279": "Scotland",
            "281": "Spain",
            "284": "Sweden",
            "285": "Grenada",
            "287": "Switzerland",
            "288": "Ukraine",
            "289": "Barbados",
            "291": "United States",
            "292": "Uzbekistan",
            "293": "Vietnam",
            "294": "Wales",
            "296": "Yugoslavia",
            "308": "Anguilla",
            "315": "Sri Lanka",
            "318": "China",
            "324": "DPR of Korea",
            "327": "Philippines",
            "330": "Taiwan",
            "336": "Hong Kong",
            "339": "Japan",
            "348": "South Korea",
            "354": "India",
            "375": "Mariana Islands",
            "376": "Thailand",
            "378": "Pakistan",
            "379": "South Sudan",
            "386": "Macau",
            "387": "Nepal",
            "390": "European Russia",
            "400": "Algeria",
            "401": "Angola",
            "404": "Burundi",
            "406": "Cameroon",
            "408": "Central African Republic",
            "409": "Cape Verde",
            "410": "Chad",
            "412": "Comoros",
            "414": "Democratic Republic of the Congo",
            "420": "Djibouti",
            "422": "Egypt",
            "424": "Eritrea",
            "428": "Ethiopia",
            "430": "Gabon",
            "432": "Gambia",
            "434": "Ghana",
            "436": "Kenya",
            "438": "Lesotho",
            "440": "Liberia",
            "442": "Libya",
            "444": "Madagascar",
            "446": "Malawi",
            "450": "Mali",
            "452": "Mauritania",
            "453": "Morocco",
            "454": "Mozambique",
            "456": "Niger",
            "458": "Nigeria",
            "462": "Reunion",
            "464": "Rwanda",
            "466": "Senegal",
            "468": "Seychelles",
            "470": "Sierra Leone",
            "474": "Somalia",
            "478": "South Africa",
            "480": "Sudan",
            "482": "Eswatini",
            "483": "Tanzania",
            "489": "Tonga",
            "497": "Croatia",
            "499": "Slovenia",
            "501": "Bosnia-Herzegovina",
            "502": "North Macedonia",
            "503": "Czech Republic",
            "504": "Slovakia",
            "508": "Samoa",
            "516": "St. Barthelemy",
            "517": "Clipperton Island",
            "518": "St. Martin",
            "520": "Bonaire",
        },
        "entities": {
            "1": "NA", "3": "AS", "4": "AF", "5": "EU", "6": "NA", "7": "EU", "9": "OC",
            "10": "AF", "11": "AS", "12": "NA", "13": "AN", "14": "AS", "15": "AS",
            "16": "OC", "17": "NA", "18": "AS", "20": "OC", "21": "EU", "22": "OC",
            "24": "AF", "29": "AF", "31": "OC", "32": "AF", "33": "AF", "34": "OC",
            "35": "OC", "36": "NA", "37": "NA", "38": "OC", "40": "EU", "41": "AF",
            "43": "NA", "45": "EU", "46": "OC", "47": "SA", "48": "OC", "49": "AF",
            "50": "NA", "51": "AF", "53": "AF", "54": "EU", "56": "SA", "60": "NA",
            "62": "NA", "63": "SA", "64": "NA", "65": "NA", "66": "NA", "69": "NA",
            "70": "NA", "71": "SA", "72": "NA", "74": "NA", "75": "AS", "76": "NA",
            "77": "NA", "78": "NA", "79": "NA", "80": "NA", "82": "NA", "84": "NA",
            "86": "NA", "88": "NA", "89": "NA", "90": "SA", "91": "SA", "94": "NA",
            "95": "NA", "96": "NA", "97": "NA", "98": "NA", "99": "AF", "100": "SA",
            "103": "OC", "104": "SA", "105": "NA", "106": "EU", "107": "AF", "108": "SA",
            "109": "AF", "110": "OC", "111": "AF", "112": "SA", "114": "EU", "116": "SA",
            "117": "EU", "118": "EU", "120": "SA", "122": "EU", "123": "OC", "124": "AF",
            "125": "SA", "126": "EU", "129": "SA", "130": "AS", "131": "AF", "132": "SA",
            "133": "OC", "135": "AS", "136": "SA", "137": "AS", "138": "OC", "140": "SA",
            "141": "SA", "142": "AS", "143": "AS", "144": "SA", "145": "EU", "146": "EU",
            "147": "OC", "148": "SA", "149": "EU", "150": "OC", "152": "AS", "153": "OC",
            "157": "OC", "158": "OC", "159": "AS", "160": "OC", "161": "SA", "162": "OC",
            "163": "OC", "165": "AF", "166": "OC", "167": "EU", "168": "OC", "169": "AF",
            "170": "OC", "171": "OC", "172": "OC", "173": "OC", "174": "OC", "175": "OC",
            "176": "OC", "177": "OC", "179": "EU", "180": "EU", "181": "AF", "182": "NA",
            "185": "OC", "187": "AF", "188": "OC", "189": "OC", "190": "OC", "191": "OC",
            "192": "AS", "195": "AF", "197": "OC", "199": "AN", "201": "AF", "202": "NA",
            "203": "EU", "204": "NA", "205": "AF", "206": "EU", "207": "AF", "209": "EU",
            "211": "NA", "212": "EU", "213": "NA", "214": "EU", "215": "AS", "216": "NA",
            "217": "SA", "219": "AF", "221": "EU", "222": "EU", "223": "EU", "224": "EU",
            "225": "EU", "227": "EU", "230": "EU", "232": "AF", "233": "EU", "235": "SA",
            "236": "EU", "237": "NA", "238": "SA", "239": "EU", "240": "SA", "241": "SA",
            "242": "EU", "245": "EU", "246": "EU", "248": "EU", "249": "NA", "250": "AF",
            "251": "EU", "252": "NA", "253": "SA", "254": "EU", "256": "AF", "257": "EU",
            "259": "EU", "260": "EU", "262": "AS", "263": "EU", "265": "EU", "266": "EU",
            "269": "EU", "270": "OC", "272": "EU", "273": "SA", "274": "AF", "275": "EU",
            "276": "AF", "277": "NA", "278": "EU", "279": "EU", "280": "AS", "281": "EU",
            "282": "OC", "283": "AS", "284": "EU", "285": "NA", "286": "AF", "287": "EU",
            "288": "EU", "289": "NA", "291": "NA", "292": "AS", "293": "AS", "294": "EU",
            "295": "EU", "296": "EU", "297": "OC", "298": "OC", "299": "AS", "301": "OC",
            "302": "AF", "303": "OC", "304": "AS", "305": "AS", "306": "AS", "308": "NA",
            "309": "AS", "312": "AS", "315": "AS", "318": "AS", "321": "AS", "324": "AS",
            "327": "OC", "330": "AS", "333": "AS", "336": "AS", "339": "AS", "342": "AS",
            "344": "AS", "345": "OC", "348": "AS", "354": "AS", "363": "AS", "369": "AS",
            "370": "AS", "372": "AS", "375": "OC", "376": "AS", "378": "AS", "379": "AF",
            "381": "AS", "382": "AF", "384": "AS", "386": "AS", "387": "AS", "390": "EU",
            "391": "AS", "400": "AF", "401": "AF", "402": "AF", "404": "AF", "406": "AF",
            "408": "AF", "409": "AF", "410": "AF", "411": "AF", "412": "AF", "414": "AF",
            "416": "AF", "420": "AF", "422": "AF", "424": "AF", "428": "AF", "430": "AF",
            "432": "AF", "434": "AF", "436": "AF", "438": "AF", "440": "AF", "442": "AF",
            "444": "AF", "446": "AF", "450": "AF", "452": "AF", "453": "AF", "454": "AF",
            "456": "AF", "458": "AF", "460": "OC", "462": "AF", "464": "AF", "466": "AF",
            "468": "AF", "470": "AF", "474": "AF", "478": "AF", "480": "AF", "482": "AF",
            "483": "AF", "489": "OC", "490": "OC", "492": "AS", "497": "EU", "499": "EU",
            "501": "EU", "502": "EU", "503": "EU", "504": "EU", "505": "AS", "506": "AS",
            "507": "OC", "508": "OC", "509": "OC", "510": "AS", "511": "OC", "512": "OC",
            "513": "OC", "514": "EU", "515": "OC", "516": "NA", "517": "SA", "518": "NA",
            "519": "NA", "520": "SA", "521": "AF", "522": "EU"
        }
    }
