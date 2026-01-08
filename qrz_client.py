"""
QRZ Logbook API client for fetching QSO data.
"""

import re
import time
import httpx
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


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
    use a smarter split that only splits on & followed by KEY=.

    Args:
        text: Raw response text

    Returns:
        Dictionary of response fields
    """
    import html
    # First decode HTML entities
    decoded = html.unescape(text)

    # Known QRZ response keys
    known_keys = ['RESULT', 'REASON', 'COUNT', 'ADIF', 'LOGID', 'LOGIDS', 'DATA']

    result = {}
    # Find each known key and extract its value
    for key in known_keys:
        pattern = f"{key}="
        idx = decoded.find(pattern)
        if idx != -1:
            # Find where the value ends (at next known key or end)
            value_start = idx + len(pattern)
            value_end = len(decoded)
            for other_key in known_keys:
                other_pattern = f"&{other_key}="
                other_idx = decoded.find(other_pattern, value_start)
                if other_idx != -1 and other_idx < value_end:
                    value_end = other_idx
            result[key] = decoded[value_start:value_end].strip()

    return result


def parse_qso_data(raw_qso: Dict[str, str]) -> Optional[QSOData]:
    """
    Convert raw ADIF dictionary to QSOData object.

    Args:
        raw_qso: Dictionary from parse_adif

    Returns:
        QSOData object or None if required fields missing
    """
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

    return QSOData(
        dx_callsign=call.upper(),
        qso_datetime=qso_datetime,
        band=raw_qso.get("BAND"),
        mode=raw_qso.get("MODE"),
        tx_power=tx_power,
        my_dxcc=my_dxcc,
        my_grid=raw_qso.get("MY_GRIDSQUARE"),
        my_sig_info=raw_qso.get("MY_SIG_INFO"),
        dx_dxcc=dx_dxcc,
        dx_grid=raw_qso.get("GRIDSQUARE"),
        dx_sig_info=raw_qso.get("SIG_INFO"),
        is_confirmed=is_confirmed,
        qrz_logid=raw_qso.get("APP_QRZLOG_LOGID"),
    )


async def fetch_qsos(api_key: str, confirmed_only: bool = True) -> List[QSOData]:
    """
    Fetch all QSOs from QRZ Logbook API.

    Args:
        api_key: User's QRZ Logbook API key
        confirmed_only: If True, only fetch confirmed QSOs

    Returns:
        List of QSOData objects

    Raises:
        QRZAPIError: If API returns an error or authentication fails
    """
    all_qsos = []
    last_logid = 0

    async with httpx.AsyncClient() as client:
        while True:
            options = ["TYPE:ADIF", "MAX:250", f"AFTERLOGID:{last_logid}"]
            if confirmed_only:
                options.append("STATUS:CONFIRMED")

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
                reason = result.get("REASON", "Unknown error")
                # "no result" is not an error, just no more data
                if "no result" in reason.lower():
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
