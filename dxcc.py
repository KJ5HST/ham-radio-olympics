"""
DXCC entity to continent mapping.
"""

import json
import os
import re
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


# Callsign prefix to continent mapping
# This is used as a fallback when dx_dxcc is not available
# Covers major ITU call sign prefixes
_PREFIX_TO_CONTINENT = {
    # North America (NA) - USA
    "W": "NA", "K": "NA", "N": "NA", "AA": "NA", "AB": "NA", "AC": "NA", "AD": "NA",
    "AE": "NA", "AF": "NA", "AG": "NA", "AH": "NA", "AI": "NA", "AJ": "NA", "AK": "NA",
    "AL": "NA", "KA": "NA", "KB": "NA", "KC": "NA", "KD": "NA", "KE": "NA", "KF": "NA",
    "KG": "NA", "KH": "NA", "KI": "NA", "KJ": "NA", "KK": "NA", "KL": "NA", "KM": "NA",
    "KN": "NA", "KO": "NA", "KP": "NA", "KQ": "NA", "KR": "NA", "KS": "NA", "KT": "NA",
    "KU": "NA", "KV": "NA", "KW": "NA", "KX": "NA", "KY": "NA", "KZ": "NA",
    "NA": "NA", "NB": "NA", "NC": "NA", "ND": "NA", "NE": "NA", "NF": "NA", "NG": "NA",
    "NH": "NA", "NI": "NA", "NJ": "NA", "NK": "NA", "NL": "NA", "NM": "NA", "NN": "NA",
    "NO": "NA", "NP": "NA", "NQ": "NA", "NR": "NA", "NS": "NA", "NT": "NA", "NU": "NA",
    "NV": "NA", "NW": "NA", "NX": "NA", "NY": "NA", "NZ": "NA",
    "WA": "NA", "WB": "NA", "WC": "NA", "WD": "NA", "WE": "NA", "WF": "NA", "WG": "NA",
    "WH": "NA", "WI": "NA", "WJ": "NA", "WK": "NA", "WL": "NA", "WM": "NA", "WN": "NA",
    "WO": "NA", "WP": "NA", "WQ": "NA", "WR": "NA", "WS": "NA", "WT": "NA", "WU": "NA",
    "WV": "NA", "WW": "NA", "WX": "NA", "WY": "NA", "WZ": "NA",
    # Canada
    "VA": "NA", "VB": "NA", "VC": "NA", "VD": "NA", "VE": "NA", "VG": "NA", "VO": "NA",
    "VX": "NA", "VY": "NA", "CY": "NA", "CF": "NA", "CG": "NA", "CH": "NA", "CI": "NA",
    "CJ": "NA", "CK": "NA", "XJ": "NA", "XK": "NA", "XL": "NA", "XM": "NA", "XN": "NA",
    "XO": "NA",
    # Mexico
    "XA": "NA", "XB": "NA", "XC": "NA", "XD": "NA", "XE": "NA", "XF": "NA",
    "4A": "NA", "4B": "NA", "4C": "NA", "6D": "NA", "6E": "NA", "6F": "NA", "6G": "NA",
    "6H": "NA", "6I": "NA", "6J": "NA",
    # Caribbean/Central America
    "TI": "NA", "TE": "NA", "TG": "NA", "HR": "NA", "HP": "NA", "HQ": "NA",
    "YN": "NA", "YS": "NA", "V3": "NA", "8P": "NA", "J3": "NA", "J6": "NA",
    "J7": "NA", "J8": "NA", "VP2": "NA", "VP5": "NA", "VP9": "NA", "ZF": "NA",
    "6Y": "NA", "HI": "NA", "HH": "NA", "CO": "NA", "CM": "NA", "PJ": "NA",

    # South America (SA)
    "LU": "SA", "LO": "SA", "LP": "SA", "LQ": "SA", "LR": "SA", "LS": "SA", "LT": "SA",
    "LV": "SA", "LW": "SA", "AY": "SA", "AZ": "SA", "L2": "SA", "L3": "SA", "L4": "SA",
    "L5": "SA", "L6": "SA", "L7": "SA", "L8": "SA", "L9": "SA",
    "PP": "SA", "PQ": "SA", "PR": "SA", "PS": "SA", "PT": "SA", "PU": "SA", "PV": "SA",
    "PW": "SA", "PX": "SA", "PY": "SA", "ZV": "SA", "ZW": "SA", "ZX": "SA", "ZY": "SA",
    "ZZ": "SA",
    "CE": "SA", "CA": "SA", "CB": "SA", "CC": "SA", "CD": "SA", "XQ": "SA", "XR": "SA",
    "3G": "SA",
    "CP": "SA", "HC": "SA", "HD": "SA", "OA": "SA", "OB": "SA", "OC": "SA",
    "CX": "SA", "CV": "SA", "YV": "SA", "YW": "SA", "YX": "SA", "YY": "SA",
    "ZP": "SA", "9Y": "SA", "9Z": "SA", "P4": "SA", "HK": "SA", "HJ": "SA",
    "8R": "SA",

    # Europe (EU)
    "G": "EU", "M": "EU", "2E": "EU", "2D": "EU", "2I": "EU", "2M": "EU", "2W": "EU",
    "GW": "EU", "GD": "EU", "GI": "EU", "GM": "EU", "GU": "EU", "GJ": "EU",
    "MW": "EU", "MD": "EU", "MI": "EU", "MM": "EU", "MU": "EU", "MJ": "EU",
    "DL": "EU", "DA": "EU", "DB": "EU", "DC": "EU", "DD": "EU", "DE": "EU", "DF": "EU",
    "DG": "EU", "DH": "EU", "DI": "EU", "DJ": "EU", "DK": "EU", "DM": "EU", "DN": "EU",
    "DO": "EU", "DP": "EU", "DQ": "EU", "DR": "EU",
    "F": "EU", "TM": "EU", "TO": "EU", "TP": "EU", "TQ": "EU", "TV": "EU",
    "I": "EU", "IA": "EU", "IB": "EU", "IC": "EU", "ID": "EU", "IE": "EU", "IF": "EU",
    "IG": "EU", "IH": "EU", "II": "EU", "IJ": "EU", "IK": "EU", "IL": "EU", "IM": "EU",
    "IN": "EU", "IO": "EU", "IP": "EU", "IQ": "EU", "IR": "EU", "IS": "EU", "IT": "EU",
    "IU": "EU", "IV": "EU", "IW": "EU", "IX": "EU", "IY": "EU", "IZ": "EU",
    "EA": "EU", "EB": "EU", "EC": "EU", "ED": "EU", "EE": "EU", "EF": "EU", "EG": "EU",
    "EH": "EU", "AM": "EU", "AN": "EU", "AO": "EU",
    "PA": "EU", "PB": "EU", "PC": "EU", "PD": "EU", "PE": "EU", "PF": "EU", "PG": "EU",
    "PH": "EU", "PI": "EU",
    "ON": "EU", "OO": "EU", "OP": "EU", "OQ": "EU", "OR": "EU", "OS": "EU", "OT": "EU",
    "SM": "EU", "SA": "EU", "SB": "EU", "SC": "EU", "SD": "EU", "SE": "EU", "SF": "EU",
    "SG": "EU", "SH": "EU", "SI": "EU", "SJ": "EU", "SK": "EU", "SL": "EU", "7S": "EU",
    "8S": "EU",
    "LA": "EU", "LB": "EU", "LC": "EU", "LD": "EU", "LE": "EU", "LF": "EU", "LG": "EU",
    "LH": "EU", "LI": "EU", "LJ": "EU", "LK": "EU", "LL": "EU", "LM": "EU", "LN": "EU",
    "OZ": "EU", "OU": "EU", "OV": "EU", "OW": "EU", "OX": "EU", "XP": "EU", "5P": "EU",
    "5Q": "EU",
    "OH": "EU", "OG": "EU", "OI": "EU", "OF": "EU",
    "SP": "EU", "SQ": "EU", "SR": "EU", "SN": "EU", "SO": "EU", "3Z": "EU", "HF": "EU",
    "OE": "EU", "HB": "EU", "HB0": "EU",
    "OK": "EU", "OL": "EU", "OM": "EU",
    "HA": "EU", "HG": "EU",
    "LZ": "EU", "YO": "EU", "YP": "EU", "YQ": "EU", "YR": "EU",
    "YU": "EU", "YT": "EU", "YZ": "EU", "4N": "EU", "4O": "EU",
    "9A": "EU", "S5": "EU", "E7": "EU", "Z3": "EU",
    "SV": "EU", "SW": "EU", "SX": "EU", "SY": "EU", "SZ": "EU", "J4": "EU",
    "CT": "EU", "CS": "EU", "CQ": "EU", "CR": "EU",
    "EI": "EU", "EJ": "EU",
    "TF": "EU", "OY": "EU",
    "LX": "EU", "T7": "EU", "3A": "EU", "C3": "EU", "HV": "EU",
    "9H": "EU", "TA": "EU", "TB": "EU", "TC": "EU",
    "UA": "EU", "UB": "EU", "UC": "EU", "UD": "EU", "UE": "EU", "UF": "EU", "UG": "EU",
    "UH": "EU", "UI": "EU", "RA": "EU", "R1": "EU", "R2": "EU", "R3": "EU", "R4": "EU",
    "R5": "EU", "R6": "EU", "R7": "EU", "R8": "EU", "R9": "EU",
    "UR": "EU", "US": "EU", "UT": "EU", "UU": "EU", "UV": "EU", "UW": "EU", "UX": "EU",
    "UY": "EU", "UZ": "EU", "EM": "EU", "EN": "EU", "EO": "EU",
    "LY": "EU", "YL": "EU", "ES": "EU",
    "ER": "EU", "EU": "EU", "EV": "EU", "EW": "EU",

    # Asia (AS)
    "JA": "AS", "JB": "AS", "JC": "AS", "JD": "AS", "JE": "AS", "JF": "AS", "JG": "AS",
    "JH": "AS", "JI": "AS", "JJ": "AS", "JK": "AS", "JL": "AS", "JM": "AS", "JN": "AS",
    "JO": "AS", "JP": "AS", "JQ": "AS", "JR": "AS", "JS": "AS", "7J": "AS", "7K": "AS",
    "7L": "AS", "7M": "AS", "7N": "AS", "8J": "AS", "8K": "AS", "8L": "AS", "8M": "AS",
    "8N": "AS",
    "HL": "AS", "HM": "AS", "6K": "AS", "6L": "AS", "6M": "AS", "6N": "AS", "DS": "AS",
    "DT": "AS",
    "BA": "AS", "BB": "AS", "BC": "AS", "BD": "AS", "BE": "AS", "BF": "AS", "BG": "AS",
    "BH": "AS", "BI": "AS", "BJ": "AS", "BK": "AS", "BL": "AS", "BM": "AS", "BN": "AS",
    "BO": "AS", "BP": "AS", "BQ": "AS", "BR": "AS", "BS": "AS", "BT": "AS", "BU": "AS",
    "BV": "AS", "BW": "AS", "BX": "AS", "BY": "AS", "BZ": "AS", "XS": "AS", "3H": "AS",
    "3I": "AS", "3J": "AS", "3K": "AS", "3L": "AS", "3M": "AS", "3N": "AS", "3O": "AS",
    "3P": "AS", "3Q": "AS", "3R": "AS", "3S": "AS", "3T": "AS", "3U": "AS",
    "VU": "AS", "AT": "AS", "AU": "AS", "AV": "AS", "AW": "AS",
    "VR": "AS", "XX": "AS",
    "HS": "AS", "E2": "AS",
    "9M": "AS", "9W": "AS",
    "9V": "AS",
    "DU": "AS", "DV": "AS", "DW": "AS", "DX": "AS", "DY": "AS", "DZ": "AS", "4D": "AS",
    "4E": "AS", "4F": "AS", "4G": "AS", "4H": "AS", "4I": "AS",
    "YB": "AS", "YC": "AS", "YD": "AS", "YE": "AS", "YF": "AS", "YG": "AS", "YH": "AS",
    "7A": "AS", "7B": "AS", "7C": "AS", "7D": "AS", "7E": "AS", "7F": "AS", "7G": "AS",
    "7H": "AS", "7I": "AS", "8A": "AS", "8B": "AS", "8C": "AS", "8D": "AS", "8E": "AS",
    "8F": "AS", "8G": "AS", "8H": "AS", "8I": "AS",
    "AP": "AS", "6P": "AS", "6Q": "AS", "6R": "AS", "6S": "AS",
    "A4": "AS", "A5": "AS", "A6": "AS", "A7": "AS", "A9": "AS",
    "HZ": "AS", "7Z": "AS", "8Z": "AS",
    "EP": "AS", "EQ": "AS", "9K": "AS", "OD": "AS", "JY": "AS", "YI": "AS", "YK": "AS",
    "4X": "AS", "4Z": "AS",
    "EK": "AS", "4J": "AS", "4K": "AS", "EX": "AS", "EY": "AS", "EZ": "AS",
    "UK": "AS", "UN": "AS", "UJ": "AS", "UL": "AS", "UM": "AS",
    "XV": "AS", "XU": "AS", "XW": "AS", "XY": "AS", "XZ": "AS", "9N": "AS",
    "4S": "AS", "S2": "AS",
    "8Q": "AS",
    "UA0": "AS", "UA9": "AS", "R0": "AS",

    # Oceania (OC)
    "VK": "OC", "AX": "OC",
    "ZL": "OC", "ZK": "OC", "ZM": "OC",
    "KH6": "OC",
    "FK": "OC", "TX": "OC",
    "YJ": "OC", "3D2": "OC", "5W": "OC", "A3": "OC", "T2": "OC", "T3": "OC",
    "V7": "OC", "V6": "OC", "T8": "OC", "KC6": "OC", "KX6": "OC",
    "P2": "OC", "KG6": "OC",
    "FO": "OC", "E5": "OC", "ZK1": "OC", "ZK2": "OC",

    # Africa (AF)
    "ZS": "AF", "ZR": "AF", "ZT": "AF", "ZU": "AF", "V5": "AF", "A2": "AF",
    "7P": "AF", "7Q": "AF", "3DA": "AF", "3D0": "AF", "9J": "AF",
    "5H": "AF", "5I": "AF", "5X": "AF", "5Z": "AF", "ET": "AF", "E3": "AF",
    "ST": "AF", "6U": "AF", "9U": "AF", "9X": "AF",
    "SU": "AF", "SS": "AF", "6O": "AF",
    "CN": "AF", "5A": "AF", "7X": "AF", "TS": "AF", "3V": "AF",
    "5N": "AF", "5O": "AF", "EL": "AF", "TU": "AF", "TY": "AF", "TZ": "AF",
    "XT": "AF", "6V": "AF", "6W": "AF", "C5": "AF", "J5": "AF", "D4": "AF",
    "9G": "AF", "9L": "AF", "5T": "AF", "5U": "AF", "5V": "AF",
    "TR": "AF", "TN": "AF", "TT": "AF", "TJ": "AF", "TL": "AF", "9Q": "AF",
    "D2": "AF", "D3": "AF", "CT3": "AF", "S9": "AF", "3C": "AF",
    "5R": "AF", "5S": "AF", "3B": "AF", "FR": "AF", "FH": "AF", "S7": "AF", "D6": "AF",

    # Antarctica (AN)
    "KC4": "AN", "DP0": "AN", "VP8": "AN", "RI1AN": "AN", "CE9": "AN", "ZL5": "AN",
    "VK0": "AN", "8J1": "AN",
}


def get_continent_from_callsign(callsign: str) -> Optional[str]:
    """
    Derive continent from callsign prefix.

    This is a fallback when dx_dxcc is not available.
    Uses ITU callsign prefix assignments.

    Args:
        callsign: The amateur radio callsign

    Returns:
        Continent code (AF, AN, AS, EU, NA, OC, SA) or None if not determinable
    """
    if not callsign:
        return None

    callsign = callsign.upper().strip()

    # Remove any suffix (like /P, /M, /QRP)
    callsign = callsign.split("/")[0]

    if not callsign:
        return None

    # Try longest prefix matches first (up to 5 chars), then shorter
    for length in range(min(5, len(callsign)), 0, -1):
        prefix = callsign[:length]
        if prefix in _PREFIX_TO_CONTINENT:
            return _PREFIX_TO_CONTINENT[prefix]

    # Try single character prefix (last resort)
    first_char = callsign[0]
    if first_char in _PREFIX_TO_CONTINENT:
        return _PREFIX_TO_CONTINENT[first_char]

    return None
