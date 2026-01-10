"""
LoTW (Logbook of the World) client for fetching QSO data.
"""

import httpx
from typing import List, Optional
from datetime import datetime
from qrz_client import QSOData, parse_adif


LOTW_REPORT_URL = "https://lotw.arrl.org/lotwuser/lotwreport.adi"
USER_AGENT = "HamRadioOlympics/1.0 (github.com/ham-radio-olympics)"


class LoTWError(Exception):
    """Exception for LoTW API errors."""
    pass


async def verify_lotw_credentials(username: str, password: str, expected_callsign: str = None) -> bool:
    """
    Verify LoTW credentials by attempting to fetch a minimal report.

    Args:
        username: LoTW username
        password: LoTW password
        expected_callsign: If provided, verify the account owns this callsign

    Returns:
        True if credentials are valid (and callsign matches if provided)
    """
    async with httpx.AsyncClient() as client:
        params = {
            "login": username,
            "password": password,
            "qso_query": "1",
            "qso_qsl": "yes",  # Only confirmed QSOs
            "qso_qsldetail": "yes",
            "qso_mydetail": "yes",
            # Limit to 1 record just to verify credentials
            "qso_startdate": "2099-01-01",  # Future date = no records, but validates auth
        }

        try:
            response = await client.get(
                LOTW_REPORT_URL,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=30.0,
                follow_redirects=True,
            )
        except httpx.HTTPError:
            return False

        text = response.text

        # LoTW returns specific error messages for auth failures
        if "Username/password incorrect" in text:
            return False
        if "not found in lotw" in text.lower():
            return False

        # If we got ADIF header, credentials are valid
        if "<PROGRAMID:" in text.upper() or "<EOH>" in text.upper() or "ARRL Logbook" in text:
            # If expected_callsign provided, verify ownership
            if expected_callsign:
                # The username for LoTW is typically the callsign
                if username.upper() != expected_callsign.upper():
                    return False
            return True

        return False


async def fetch_lotw_qsos(
    username: str,
    password: str,
    confirmed_only: bool = True,
    start_date: str = None
) -> List[QSOData]:
    """
    Fetch QSOs from LoTW.

    Args:
        username: LoTW username
        password: LoTW password
        confirmed_only: If True, only fetch confirmed QSOs
        start_date: Optional start date (YYYY-MM-DD) to limit results

    Returns:
        List of QSOData objects

    Raises:
        LoTWError: If authentication fails or API returns an error
    """
    async with httpx.AsyncClient() as client:
        params = {
            "login": username,
            "password": password,
            "qso_query": "1",
            "qso_qsldetail": "yes",
            "qso_mydetail": "yes",
        }

        if confirmed_only:
            params["qso_qsl"] = "yes"

        if start_date:
            params["qso_startdate"] = start_date

        try:
            response = await client.get(
                LOTW_REPORT_URL,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=60.0,
                follow_redirects=True,
            )
        except httpx.HTTPError as e:
            raise LoTWError(f"HTTP error: {e}")

        text = response.text

        # Check for auth errors
        if "Username/password incorrect" in text:
            raise LoTWError("Authentication failed - invalid username or password")
        if "not found in lotw" in text.lower():
            raise LoTWError("Account not found in LoTW")

        # Parse ADIF data
        raw_qsos = parse_adif(text)

        qsos = []
        for raw_qso in raw_qsos:
            qso = parse_lotw_qso(raw_qso)
            if qso:
                qsos.append(qso)

        return qsos


def parse_lotw_qso(raw_qso: dict) -> Optional[QSOData]:
    """
    Convert raw ADIF dictionary from LoTW to QSOData object.

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

    # LoTW QSOs in our query are confirmed by definition (qso_qsl=yes)
    is_confirmed = True

    # Also check QSL_RCVD field if present
    qsl_rcvd = raw_qso.get("QSL_RCVD", "").upper()
    lotw_qsl_rcvd = raw_qso.get("LOTW_QSL_RCVD", "").upper()
    if qsl_rcvd == "Y" or lotw_qsl_rcvd == "Y":
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
        qrz_logid=None,  # LoTW doesn't use QRZ log IDs
    )
