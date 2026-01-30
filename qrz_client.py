"""
QRZ Logbook API client for fetching QSO data.
"""

import logging
import re
import time
import httpx
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


QRZ_API_URL = "https://logbook.qrz.com/api"
USER_AGENT = "HamRadioOlympics/1.0 (github.com/ham-radio-olympics)"


@dataclass
class QSOData:
    """Parsed QSO data from QRZ."""
    dx_callsign: str
    qso_datetime: datetime
    band: Optional[str]
    mode: Optional[str]
    tx_power: Optional[float]
    my_dxcc: Optional[int]
    my_grid: Optional[str]
    my_sig_info: Optional[str]
    dx_dxcc: Optional[int]
    dx_grid: Optional[str]
    dx_sig_info: Optional[str]
    is_confirmed: bool
    qrz_logid: Optional[str]


# POTA park reference pattern: 1-3 letter country code, dash, 3+ digits
# Examples: US-0303, K-0001, VE-1234, G-0001, DL-0001
POTA_PATTERN = re.compile(r'\b([A-Z]{1,3}-\d{3,})\b', re.IGNORECASE)


def _normalize_park_id(park_id: Optional[str]) -> Optional[str]:
    """
    Normalize a POTA park ID to standard format (zero-padded to 4 digits).

    Args:
        park_id: Raw park ID (e.g., "US-303")

    Returns:
        Normalized park ID (e.g., "US-0303") or None if invalid
    """
    if not park_id:
        return None

    park_id = park_id.upper().strip()
    match = POTA_PATTERN.match(park_id)

    if not match:
        return None

    # Extract prefix and number
    parts = park_id.split('-', 1)
    if len(parts) != 2:
        return None

    prefix = parts[0]
    number = parts[1]

    # Zero-pad to at least 4 digits
    number = number.zfill(4)

    return f"{prefix}-{number}"


def _extract_pota_from_comment(comment: str) -> Optional[str]:
    """
    Extract and normalize POTA park reference from a comment field.

    Args:
        comment: The comment text to search

    Returns:
        Normalized park reference (e.g., "US-0303") if found, None otherwise
    """
    if not comment:
        return None

    match = POTA_PATTERN.search(comment)
    if match:
        return _normalize_park_id(match.group(1))
    return None


def parse_adif(adif_string: str) -> List[Dict[str, str]]:
    """
    Parse ADIF format data into a list of QSO dictionaries.

    ADIF format: <FIELD:LENGTH>value<FIELD:LENGTH>value...<EOR>

    Args:
        adif_string: Raw ADIF data string

    Returns:
        List of dictionaries, each containing QSO field/value pairs
    """
    qsos = []
    # Split by <EOR> case-insensitively
    records = re.split(r'<eor>', adif_string, flags=re.IGNORECASE)

    for record in records:
        if not record.strip():
            continue

        qso = {}
        # Match <FIELD:LENGTH>value patterns
        pattern = r'<(\w+):(\d+)(?::[^>]*)?>([^<]*)'

        for match in re.finditer(pattern, record, re.IGNORECASE):
            field_name = match.group(1).upper()
            length = int(match.group(2))
            value = match.group(3)[:length].strip()
            if value:
                qso[field_name] = value

        if qso:
            qsos.append(qso)

    return qsos


def parse_qrz_response(text: str) -> Dict[str, str]:
    """
    Parse QRZ API response format (key=value&key=value).

    QRZ uses & as a field separator, but also encodes < > as &lt; &gt;
    in the ADIF data. We need to decode HTML entities first, then
    parse carefully to handle ADIF fields that may contain & characters.

    Args:
        text: Raw response text

    Returns:
        Dictionary of response fields
    """
    import html
    # First decode HTML entities
    decoded = html.unescape(text)

    result = {}

    # Special handling for ADIF field which can contain & in its content
    adif_marker = "ADIF="
    adif_idx = decoded.find(adif_marker)
    adif_content = ""
    non_adif_text = decoded

    if adif_idx != -1:
        # ADIF field goes to the end or until we hit a known terminal field
        adif_start = adif_idx + len(adif_marker)
        adif_content = decoded[adif_start:]
        non_adif_text = decoded[:adif_idx]
        result["ADIF"] = adif_content.strip()

    # Parse the non-ADIF portion as simple key=value pairs
    for pair in non_adif_text.split("&"):
        if "=" in pair:
            key, value = pair.split("=", 1)
            result[key.strip()] = value.strip()

    return result


import logging

logger = logging.getLogger(__name__)


def parse_qso_data(raw_qso: Dict[str, str]) -> Optional[QSOData]:
    """
    Convert raw ADIF dictionary to QSOData object.

    Args:
        raw_qso: Dictionary from parse_adif

    Returns:
        QSOData object or None if required fields missing
    """
    # Log raw ADIF fields for debugging park data issues
    # Look for any field containing 'SIG', 'POTA', 'WWFF', or 'PARK'
    sig_fields = {k: v for k, v in raw_qso.items()
                  if any(x in k.upper() for x in ['SIG', 'POTA', 'WWFF', 'PARK'])}
    if sig_fields:
        logger.debug(f"QSO {raw_qso.get('CALL')} park-related fields: {sig_fields}")

    # Required fields
    call = raw_qso.get("CALL")
    qso_date = raw_qso.get("QSO_DATE")
    time_on = raw_qso.get("TIME_ON", "0000")

    if not call or not qso_date:
        return None

    # Parse datetime
    try:
        # Handle both HHMM and HHMMSS formats
        if len(time_on) == 4:
            time_on = time_on + "00"
        dt_str = f"{qso_date}{time_on}"
        qso_datetime = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    except ValueError:
        return None

    # Parse TX power
    tx_power = None
    tx_pwr_str = raw_qso.get("TX_PWR")
    if tx_pwr_str:
        try:
            tx_power = float(tx_pwr_str)
        except ValueError:
            pass

    # Parse DXCC codes
    my_dxcc = None
    dx_dxcc = None

    my_dxcc_str = raw_qso.get("MY_DXCC")
    if my_dxcc_str:
        try:
            my_dxcc = int(my_dxcc_str)
        except ValueError:
            pass

    dx_dxcc_str = raw_qso.get("DXCC")
    if dx_dxcc_str:
        try:
            dx_dxcc = int(dx_dxcc_str)
        except ValueError:
            pass

    # Determine confirmation status
    # QRZ uses various fields to indicate confirmation
    is_confirmed = False
    qsl_rcvd = raw_qso.get("QSL_RCVD", "").upper()
    lotw_qsl_rcvd = raw_qso.get("LOTW_QSL_RCVD", "").upper()
    app_qrzlog_status = raw_qso.get("APP_QRZLOG_STATUS", "").upper()

    if qsl_rcvd == "Y" or lotw_qsl_rcvd == "Y" or app_qrzlog_status == "C":
        is_confirmed = True

    # Check multiple possible field names for POTA/SIG info
    # Standard ADIF uses SIG_INFO, but some loggers use POTA_REF or app-specific fields
    # Also check COMMENT field as fallback (some users put park refs there)
    # Normalize all park IDs to standard format (zero-padded to 4 digits)
    #
    # Field order priority:
    # 1. Standard ADIF: MY_SIG_INFO, SIG_INFO
    # 2. POTA-specific: MY_POTA_REF, POTA_REF
    # 3. WWFF-specific: MY_WWFF_REF, WWFF_REF
    # 4. App-specific fields that some loggers use
    # 5. Comments as fallback
    # Note: SIG and MY_SIG contain the program name (e.g., "POTA"), not the park reference
    # The park reference is in SIG_INFO and MY_SIG_INFO
    my_sig_info = (
        _normalize_park_id(raw_qso.get("MY_SIG_INFO")) or
        _normalize_park_id(raw_qso.get("MY_POTA_REF")) or
        _normalize_park_id(raw_qso.get("MY_WWFF_REF")) or
        _normalize_park_id(raw_qso.get("APP_POTA_MYPARKREF")) or  # App-specific
        _normalize_park_id(raw_qso.get("APP_POTA_REF")) or  # App-specific variant
        _extract_pota_from_comment(raw_qso.get("MY_COMMENT", "")) or
        _extract_pota_from_comment(raw_qso.get("NOTES", ""))  # Some loggers use NOTES
    )
    dx_sig_info = (
        _normalize_park_id(raw_qso.get("SIG_INFO")) or
        _normalize_park_id(raw_qso.get("POTA_REF")) or
        _normalize_park_id(raw_qso.get("WWFF_REF")) or
        _normalize_park_id(raw_qso.get("APP_POTA_PARKREF")) or  # App-specific
        _extract_pota_from_comment(raw_qso.get("COMMENT", "")) or
        _extract_pota_from_comment(raw_qso.get("QSO_NOTES", ""))  # Some loggers use QSO_NOTES
    )

    return QSOData(
        dx_callsign=call.upper(),
        qso_datetime=qso_datetime,
        band=raw_qso.get("BAND"),
        mode=raw_qso.get("MODE"),
        tx_power=tx_power,
        my_dxcc=my_dxcc,
        my_grid=raw_qso.get("MY_GRIDSQUARE"),
        my_sig_info=my_sig_info,
        dx_dxcc=dx_dxcc,
        dx_grid=raw_qso.get("GRIDSQUARE"),
        dx_sig_info=dx_sig_info,
        is_confirmed=is_confirmed,
        qrz_logid=raw_qso.get("APP_QRZLOG_LOGID"),
    )


async def verify_api_key(api_key: str, expected_callsign: str = None) -> bool:
    """
    Verify that a QRZ API key is valid and optionally belongs to a specific callsign.

    Args:
        api_key: User's QRZ Logbook API key
        expected_callsign: If provided, verify the API key belongs to this callsign

    Returns:
        True if the API key is valid (and matches callsign if provided), False otherwise
    """
    async with httpx.AsyncClient() as client:
        data = {
            "KEY": api_key,
            "ACTION": "STATUS",
        }

        try:
            response = await client.post(
                QRZ_API_URL,
                data=data,
                headers={"User-Agent": USER_AGENT},
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            return False

        result = parse_qrz_response(response.text)

        # Check if the API returned OK
        if result.get("RESULT") != "OK":
            return False

        # If expected_callsign provided, verify it matches the API key's owner
        if expected_callsign:
            # Try multiple fields where callsign might be returned
            api_callsign = result.get("CALLSIGN", "").upper()

            # Also check DATA field which may contain OWNER or CALLSIGN
            data_field = result.get("DATA", "")
            if not api_callsign and data_field:
                # Parse DATA field (format: NAME:VALUE,NAME:VALUE,...)
                for item in data_field.split(","):
                    if ":" in item:
                        key, value = item.split(":", 1)
                        key = key.strip().upper()
                        if key in ("CALLSIGN", "OWNER", "CALL"):
                            api_callsign = value.strip().upper()
                            break

            # If we found a callsign in the response, verify it matches
            if api_callsign:
                # Normalize callsigns for comparison (strip /P, /M, etc. suffixes for matching)
                expected_base = expected_callsign.upper().split("/")[0]
                api_base = api_callsign.split("/")[0]
                if api_base != expected_base:
                    return False
            # If no callsign in response, just verify the key is valid (RESULT=OK)
            # This is a fallback - the key works, we just can't verify ownership

        return True


async def fetch_qsos(
    api_key: str,
    confirmed_only: bool = True,
    since_date: Optional[datetime] = None,
    until_date: Optional[datetime] = None,
    after_logid: int = 0,
) -> List[QSOData]:
    """
    Fetch QSOs from QRZ Logbook API.

    Args:
        api_key: User's QRZ Logbook API key
        confirmed_only: If True, only fetch confirmed QSOs
        since_date: If provided, only fetch QSOs from this date forward (inclusive)
            NOTE: The QRZ BETWEEN filter is unreliable for some accounts.
            Prefer using after_logid for incremental syncs.
        until_date: If provided, only fetch QSOs until this date (inclusive)
        after_logid: If provided, only fetch QSOs with logid > this value.
            This is the preferred method for incremental syncs.

    Returns:
        List of QSOData objects

    Raises:
        QRZAPIError: If API returns an error or authentication fails
    """
    all_qsos = []
    last_logid = after_logid

    async with httpx.AsyncClient() as client:
        while True:
            options = ["TYPE:ADIF", "MAX:250", f"AFTERLOGID:{last_logid}"]
            if confirmed_only:
                options.append("STATUS:CONFIRMED")

            # Note: The BETWEEN filter is unreliable for some QRZ accounts
            # and may return COUNT=0 even when QSOs exist. We keep it as a
            # fallback but prefer using after_logid for incremental syncs.
            if since_date or until_date:
                start_str = since_date.strftime("%Y%m%d") if since_date else "19000101"
                end_str = until_date.strftime("%Y%m%d") if until_date else "20991231"
                options.append(f"BETWEEN:{start_str}+{end_str}")

            data = {
                "KEY": api_key,
                "ACTION": "FETCH",
                "OPTION": ",".join(options),
            }

            try:
                response = await client.post(
                    QRZ_API_URL,
                    data=data,
                    headers={"User-Agent": USER_AGENT},
                    timeout=30.0,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise QRZAPIError(f"HTTP error: {e}")

            result = parse_qrz_response(response.text)

            if result.get("RESULT") == "AUTH":
                raise QRZAPIError("Authentication failed - invalid API key")

            if result.get("RESULT") == "FAIL":
                reason = result.get("REASON", "")
                # "no result" is not an error, just no more data
                if "no result" in reason.lower():
                    break
                # COUNT=0 with RESULT=FAIL means empty logbook (no QSOs match)
                if result.get("COUNT") == "0" or reason == "":
                    logger.info(f"QRZ returned no QSOs (COUNT=0 or empty response)")
                    break
                raise QRZAPIError(f"API error: {reason}")

            adif_data = result.get("ADIF", "")
            if not adif_data:
                break

            raw_qsos = parse_adif(adif_data)
            if not raw_qsos:
                break

            for raw_qso in raw_qsos:
                qso = parse_qso_data(raw_qso)
                if qso:
                    all_qsos.append(qso)

            # Get highest logid for pagination
            max_logid = 0
            for raw_qso in raw_qsos:
                logid_str = raw_qso.get("APP_QRZLOG_LOGID", "0")
                try:
                    logid = int(logid_str)
                    if logid > max_logid:
                        max_logid = logid
                except ValueError:
                    pass

            if max_logid <= last_logid:
                break
            last_logid = max_logid

            # Rate limiting - be respectful
            time.sleep(1)

    return all_qsos


class QRZAPIError(Exception):
    """Exception for QRZ API errors."""
    pass
