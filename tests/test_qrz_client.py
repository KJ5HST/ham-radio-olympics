"""
Tests for QRZ API client and ADIF parsing.
"""

import pytest
import respx
import httpx
from datetime import datetime
from unittest.mock import patch

from qrz_client import parse_adif, parse_qrz_response, parse_qso_data, fetch_qsos, verify_api_key, QRZAPIError, QRZ_API_URL


class TestADIFParsing:
    """Test ADIF format parsing."""

    def test_parse_simple_adif(self):
        """Test parsing a simple ADIF record."""
        adif = "<CALL:5>W1ABC<QSO_DATE:8>20260115<TIME_ON:4>1423<BAND:3>20M<MODE:3>SSB<EOR>"
        result = parse_adif(adif)

        assert len(result) == 1
        assert result[0]["CALL"] == "W1ABC"
        assert result[0]["QSO_DATE"] == "20260115"
        assert result[0]["TIME_ON"] == "1423"
        assert result[0]["BAND"] == "20M"
        assert result[0]["MODE"] == "SSB"

    def test_parse_multiple_records(self):
        """Test parsing multiple ADIF records."""
        adif = """
        <CALL:5>W1ABC<QSO_DATE:8>20260115<TIME_ON:4>1423<EOR>
        <CALL:5>K2DEF<QSO_DATE:8>20260116<TIME_ON:4>0800<EOR>
        """
        result = parse_adif(adif)

        assert len(result) == 2
        assert result[0]["CALL"] == "W1ABC"
        assert result[1]["CALL"] == "K2DEF"

    def test_parse_with_all_fields(self):
        """Test parsing ADIF with all relevant fields."""
        adif = """<CALL:5>DL1AB<QSO_DATE:8>20260101<TIME_ON:6>120100<BAND:3>20M<MODE:3>SSB
        <TX_PWR:3>5.0<MY_GRIDSQUARE:4>EM12<GRIDSQUARE:4>JN58<DXCC:3>230
        <MY_SIG_INFO:6>K-4556<SIG_INFO:6>K-0001<QSL_RCVD:1>Y<EOR>"""

        result = parse_adif(adif)

        assert len(result) == 1
        qso = result[0]
        assert qso["CALL"] == "DL1AB"
        assert qso["TX_PWR"] == "5.0"
        assert qso["MY_GRIDSQUARE"] == "EM12"
        assert qso["GRIDSQUARE"] == "JN58"
        assert qso["DXCC"] == "230"
        assert qso["MY_SIG_INFO"] == "K-4556"
        assert qso["SIG_INFO"] == "K-0001"
        assert qso["QSL_RCVD"] == "Y"

    def test_parse_empty_adif(self):
        """Test parsing empty ADIF string."""
        result = parse_adif("")
        assert result == []

    def test_parse_adif_with_no_eor(self):
        """Test parsing ADIF without proper EOR marker."""
        adif = "<CALL:5>W1ABC<QSO_DATE:8>20260115"
        result = parse_adif(adif)
        # Should still parse but might be incomplete
        assert len(result) <= 1


class TestQRZResponseParsing:
    """Test QRZ API response format parsing."""

    def test_parse_success_response(self):
        """Test parsing successful QRZ response."""
        response = "RESULT=OK&COUNT=5&LOGID=12345"
        result = parse_qrz_response(response)

        assert result["RESULT"] == "OK"
        assert result["COUNT"] == "5"
        assert result["LOGID"] == "12345"

    def test_parse_fail_response(self):
        """Test parsing failed QRZ response."""
        response = "RESULT=FAIL&REASON=Invalid API key"
        result = parse_qrz_response(response)

        assert result["RESULT"] == "FAIL"
        assert result["REASON"] == "Invalid API key"

    def test_parse_response_with_adif(self):
        """Test parsing response containing ADIF data."""
        response = "RESULT=OK&ADIF=<CALL:5>W1ABC<EOR>"
        result = parse_qrz_response(response)

        assert result["RESULT"] == "OK"
        assert "<CALL:5>W1ABC" in result["ADIF"]


class TestQSODataParsing:
    """Test converting raw ADIF to QSOData objects."""

    def test_parse_confirmed_qso(self):
        """Test parsing a confirmed QSO."""
        raw = {
            "CALL": "DL1ABC",
            "QSO_DATE": "20260101",
            "TIME_ON": "1201",
            "BAND": "20M",
            "MODE": "SSB",
            "TX_PWR": "5.0",
            "MY_GRIDSQUARE": "EM12",
            "GRIDSQUARE": "JN58",
            "DXCC": "230",
            "QSL_RCVD": "Y",
        }

        result = parse_qso_data(raw)

        assert result is not None
        assert result.dx_callsign == "DL1ABC"
        assert result.qso_datetime == datetime(2026, 1, 1, 12, 1, 0)
        assert result.band == "20M"
        assert result.mode == "SSB"
        assert result.tx_power == 5.0
        assert result.my_grid == "EM12"
        assert result.dx_grid == "JN58"
        assert result.dx_dxcc == 230
        assert result.is_confirmed == True

    def test_parse_unconfirmed_qso(self):
        """Test parsing an unconfirmed QSO."""
        raw = {
            "CALL": "JA1XYZ",
            "QSO_DATE": "20260101",
            "TIME_ON": "1210",
            "TX_PWR": "3.0",
            "GRIDSQUARE": "PM95",
            "DXCC": "339",
            # No QSL_RCVD field
        }

        result = parse_qso_data(raw)

        assert result is not None
        assert result.dx_callsign == "JA1XYZ"
        assert result.is_confirmed == False

    def test_parse_lotw_confirmed_qso(self):
        """Test parsing QSO confirmed via LoTW."""
        raw = {
            "CALL": "K1ABC",
            "QSO_DATE": "20260115",
            "TIME_ON": "0830",
            "LOTW_QSL_RCVD": "Y",
        }

        result = parse_qso_data(raw)
        assert result.is_confirmed == True

    def test_parse_qrz_status_confirmed(self):
        """Test parsing QSO with QRZ status confirmed."""
        raw = {
            "CALL": "W2XYZ",
            "QSO_DATE": "20260115",
            "TIME_ON": "0900",
            "APP_QRZLOG_STATUS": "C",
        }

        result = parse_qso_data(raw)
        assert result.is_confirmed == True

    def test_parse_missing_call(self):
        """Test parsing QSO with missing CALL returns None."""
        raw = {
            "QSO_DATE": "20260101",
            "TIME_ON": "1200",
        }

        result = parse_qso_data(raw)
        assert result is None

    def test_parse_missing_date(self):
        """Test parsing QSO with missing date returns None."""
        raw = {
            "CALL": "W1ABC",
            "TIME_ON": "1200",
        }

        result = parse_qso_data(raw)
        assert result is None

    def test_parse_six_digit_time(self):
        """Test parsing QSO with HHMMSS time format."""
        raw = {
            "CALL": "K3ABC",
            "QSO_DATE": "20260120",
            "TIME_ON": "143025",
        }

        result = parse_qso_data(raw)
        assert result.qso_datetime == datetime(2026, 1, 20, 14, 30, 25)

    def test_parse_activate_mode_fields(self):
        """Test parsing QSO with MY_* fields for activate mode."""
        raw = {
            "CALL": "W6ABC",
            "QSO_DATE": "20260101",
            "TIME_ON": "1205",
            "MY_GRIDSQUARE": "DN15",
            "MY_SIG_INFO": "K-0001",
            "MY_DXCC": "291",
            "GRIDSQUARE": "CM87",
            "DXCC": "291",
            "QSL_RCVD": "Y",
        }

        result = parse_qso_data(raw)

        assert result.my_grid == "DN15"
        assert result.my_sig_info == "K-0001"
        assert result.my_dxcc == 291
        assert result.dx_grid == "CM87"

    def test_parse_invalid_datetime(self):
        """Test parsing QSO with invalid datetime returns None."""
        raw = {
            "CALL": "W1ABC",
            "QSO_DATE": "invalid",
            "TIME_ON": "1200",
        }
        result = parse_qso_data(raw)
        assert result is None

    def test_parse_invalid_tx_power(self):
        """Test parsing QSO with invalid TX power."""
        raw = {
            "CALL": "W1ABC",
            "QSO_DATE": "20260101",
            "TIME_ON": "1200",
            "TX_PWR": "not_a_number",
        }
        result = parse_qso_data(raw)
        assert result is not None
        assert result.tx_power is None

    def test_parse_invalid_my_dxcc(self):
        """Test parsing QSO with invalid MY_DXCC."""
        raw = {
            "CALL": "W1ABC",
            "QSO_DATE": "20260101",
            "TIME_ON": "1200",
            "MY_DXCC": "not_a_number",
        }
        result = parse_qso_data(raw)
        assert result is not None
        assert result.my_dxcc is None

    def test_parse_invalid_dxcc(self):
        """Test parsing QSO with invalid DXCC."""
        raw = {
            "CALL": "W1ABC",
            "QSO_DATE": "20260101",
            "TIME_ON": "1200",
            "DXCC": "invalid",
        }
        result = parse_qso_data(raw)
        assert result is not None
        assert result.dx_dxcc is None


class TestFetchQSOs:
    """Test async QRZ API fetching with respx mocking."""

    @pytest.mark.asyncio
    @respx.mock
    @patch("qrz_client.time.sleep")
    async def test_fetch_qsos_success(self, mock_sleep):
        """Test successful QSO fetch with single page."""
        adif_data = "<CALL:5>W1ABC<QSO_DATE:8>20260115<TIME_ON:4>1423<APP_QRZLOG_LOGID:3>100<EOR>"
        response_text = f"RESULT=OK&ADIF={adif_data}"

        # First request returns data, second returns "no result"
        respx.post(QRZ_API_URL).mock(side_effect=[
            httpx.Response(200, text=response_text),
            httpx.Response(200, text="RESULT=FAIL&REASON=no result"),
        ])

        result = await fetch_qsos("test-api-key")

        assert len(result) == 1
        assert result[0].dx_callsign == "W1ABC"

    @pytest.mark.asyncio
    @respx.mock
    @patch("qrz_client.time.sleep")
    async def test_fetch_qsos_unconfirmed(self, mock_sleep):
        """Test fetch without confirmed_only flag (covers line 189)."""
        adif_data = "<CALL:5>K2DEF<QSO_DATE:8>20260116<TIME_ON:4>0800<APP_QRZLOG_LOGID:3>200<EOR>"
        response_text = f"RESULT=OK&ADIF={adif_data}"

        respx.post(QRZ_API_URL).mock(side_effect=[
            httpx.Response(200, text=response_text),
            httpx.Response(200, text="RESULT=FAIL&REASON=no result"),
        ])

        result = await fetch_qsos("test-api-key", confirmed_only=False)

        assert len(result) == 1
        assert result[0].dx_callsign == "K2DEF"

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_http_error(self):
        """Test HTTP error handling (covers lines 205-206)."""
        respx.post(QRZ_API_URL).mock(return_value=httpx.Response(500))

        with pytest.raises(QRZAPIError) as exc:
            await fetch_qsos("test-api-key")

        assert "HTTP error" in str(exc.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_connection_error(self):
        """Test connection error handling."""
        respx.post(QRZ_API_URL).mock(side_effect=httpx.ConnectError("Connection failed"))

        with pytest.raises(QRZAPIError) as exc:
            await fetch_qsos("test-api-key")

        assert "HTTP error" in str(exc.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_auth_failure(self):
        """Test authentication failure (covers lines 210-211)."""
        respx.post(QRZ_API_URL).mock(return_value=httpx.Response(200, text="RESULT=AUTH"))

        with pytest.raises(QRZAPIError) as exc:
            await fetch_qsos("test-api-key")

        assert "Authentication failed" in str(exc.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_api_error(self):
        """Test API error with reason (covers lines 213-218)."""
        respx.post(QRZ_API_URL).mock(
            return_value=httpx.Response(200, text="RESULT=FAIL&REASON=Rate limit exceeded")
        )

        with pytest.raises(QRZAPIError) as exc:
            await fetch_qsos("test-api-key")

        assert "Rate limit exceeded" in str(exc.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_empty_logbook(self):
        """Test empty logbook returns empty list."""
        respx.post(QRZ_API_URL).mock(
            return_value=httpx.Response(200, text="RESULT=FAIL&REASON=no result")
        )

        result = await fetch_qsos("test-api-key")
        assert result == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_no_adif_data(self):
        """Test response with no ADIF data (covers lines 228-229)."""
        respx.post(QRZ_API_URL).mock(return_value=httpx.Response(200, text="RESULT=OK"))

        result = await fetch_qsos("test-api-key")

        assert result == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_empty_adif(self):
        """Test response with empty ADIF (covers lines 221-222)."""
        respx.post(QRZ_API_URL).mock(return_value=httpx.Response(200, text="RESULT=OK&ADIF="))

        result = await fetch_qsos("test-api-key")

        assert result == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_unparseable_adif(self):
        """Test response with ADIF that parses to empty list (covers lines 225-226)."""
        # ADIF data that doesn't contain any valid records
        respx.post(QRZ_API_URL).mock(return_value=httpx.Response(200, text="RESULT=OK&ADIF=garbage data with no EOR"))

        result = await fetch_qsos("test-api-key")

        assert result == []

    @pytest.mark.asyncio
    @respx.mock
    @patch("qrz_client.time.sleep")
    async def test_fetch_qsos_pagination(self, mock_sleep):
        """Test pagination with multiple pages (covers lines 234-246)."""
        # First page with logid 100
        adif_page1 = "<CALL:5>W1ABC<QSO_DATE:8>20260115<TIME_ON:4>1400<APP_QRZLOG_LOGID:3>100<EOR>"
        # Second page with logid 200
        adif_page2 = "<CALL:5>K2DEF<QSO_DATE:8>20260116<TIME_ON:4>0800<APP_QRZLOG_LOGID:3>200<EOR>"

        respx.post(QRZ_API_URL).mock(side_effect=[
            httpx.Response(200, text=f"RESULT=OK&ADIF={adif_page1}"),
            httpx.Response(200, text=f"RESULT=OK&ADIF={adif_page2}"),
            httpx.Response(200, text="RESULT=FAIL&REASON=no result"),
        ])

        result = await fetch_qsos("test-api-key")

        assert len(result) == 2
        assert result[0].dx_callsign == "W1ABC"
        assert result[1].dx_callsign == "K2DEF"

    @pytest.mark.asyncio
    @respx.mock
    @patch("qrz_client.time.sleep")
    async def test_fetch_qsos_pagination_stall(self, mock_sleep):
        """Test pagination stops when logid doesn't increase (covers line 244-245)."""
        # Same logid returned twice - should stop pagination after second request
        adif_data = "<CALL:5>W1ABC<QSO_DATE:8>20260115<TIME_ON:4>1400<APP_QRZLOG_LOGID:3>100<EOR>"

        respx.post(QRZ_API_URL).mock(side_effect=[
            httpx.Response(200, text=f"RESULT=OK&ADIF={adif_data}"),
            httpx.Response(200, text=f"RESULT=OK&ADIF={adif_data}"),
        ])

        result = await fetch_qsos("test-api-key")

        # It fetches first page, then second page returns same logid, so it stops
        # But both pages are parsed before the logid check happens
        assert len(result) == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_invalid_logid(self):
        """Test handling invalid logid in response (covers lines 241-242)."""
        adif_data = "<CALL:5>W1ABC<QSO_DATE:8>20260115<TIME_ON:4>1400<APP_QRZLOG_LOGID:7>invalid<EOR>"

        respx.post(QRZ_API_URL).mock(return_value=httpx.Response(200, text=f"RESULT=OK&ADIF={adif_data}"))

        result = await fetch_qsos("test-api-key")

        # Should parse QSO but stop pagination due to invalid logid
        assert len(result) == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_skips_invalid_qso(self):
        """Test that invalid QSOs are skipped (covers line 230)."""
        # First QSO valid, second missing required CALL
        adif_data = "<CALL:5>W1ABC<QSO_DATE:8>20260115<TIME_ON:4>1400<APP_QRZLOG_LOGID:3>100<EOR><QSO_DATE:8>20260115<TIME_ON:4>1500<APP_QRZLOG_LOGID:3>101<EOR>"

        respx.post(QRZ_API_URL).mock(side_effect=[
            httpx.Response(200, text=f"RESULT=OK&ADIF={adif_data}"),
            httpx.Response(200, text="RESULT=FAIL&REASON=no result"),
        ])

        result = await fetch_qsos("test-api-key")

        # Only the valid QSO should be returned
        assert len(result) == 1
        assert result[0].dx_callsign == "W1ABC"

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_with_date_range(self):
        """Test fetch_qsos with date range filtering."""
        from datetime import datetime
        adif_data = "<CALL:5>W1ABC<QSO_DATE:8>20260115<TIME_ON:4>1400<APP_QRZLOG_LOGID:3>100<EOR>"

        # Capture the request to verify BETWEEN option
        route = respx.post(QRZ_API_URL).mock(side_effect=[
            httpx.Response(200, text=f"RESULT=OK&ADIF={adif_data}"),
            httpx.Response(200, text="RESULT=FAIL&REASON=no result"),
        ])

        since_date = datetime(2026, 1, 1)
        until_date = datetime(2026, 1, 31)
        result = await fetch_qsos("test-api-key", since_date=since_date, until_date=until_date)

        # Verify the BETWEEN option was included in the request
        assert len(route.calls) >= 1
        request_body = route.calls[0].request.content.decode()
        assert "BETWEEN" in request_body
        assert "20260101" in request_body  # since_date
        assert "20260131" in request_body  # until_date

        assert len(result) == 1
        assert result[0].dx_callsign == "W1ABC"

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_with_only_since_date(self):
        """Test fetch_qsos with only since_date (until defaults to far future)."""
        from datetime import datetime
        adif_data = "<CALL:5>K2DEF<QSO_DATE:8>20260115<TIME_ON:4>1400<APP_QRZLOG_LOGID:3>100<EOR>"

        route = respx.post(QRZ_API_URL).mock(side_effect=[
            httpx.Response(200, text=f"RESULT=OK&ADIF={adif_data}"),
            httpx.Response(200, text="RESULT=FAIL&REASON=no result"),
        ])

        since_date = datetime(2026, 1, 1)
        result = await fetch_qsos("test-api-key", since_date=since_date)

        # Verify the BETWEEN option was included
        request_body = route.calls[0].request.content.decode()
        assert "BETWEEN" in request_body
        assert "20260101" in request_body  # since_date
        assert "20991231" in request_body  # default until_date

        assert len(result) == 1



class TestVerifyApiKey:
    """Test QRZ API key verification."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_valid_key(self):
        """Test verification of valid API key."""
        respx.post(QRZ_API_URL).mock(return_value=httpx.Response(200, text="RESULT=OK"))

        result = await verify_api_key("valid-api-key")
        assert result is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_invalid_key(self):
        """Test verification of invalid API key."""
        respx.post(QRZ_API_URL).mock(return_value=httpx.Response(200, text="RESULT=FAIL&REASON=invalid api key"))

        result = await verify_api_key("invalid-api-key")
        assert result is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_http_error(self):
        """Test verification when HTTP error occurs."""
        respx.post(QRZ_API_URL).mock(return_value=httpx.Response(500, text="Internal Server Error"))

        result = await verify_api_key("any-key")
        assert result is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_timeout(self):
        """Test verification when request times out."""
        respx.post(QRZ_API_URL).mock(side_effect=httpx.TimeoutException("timeout"))

        result = await verify_api_key("any-key")
        assert result is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_key_with_matching_callsign(self):
        """Test verification passes when callsign matches API key owner."""
        respx.post(QRZ_API_URL).mock(return_value=httpx.Response(200, text="RESULT=OK&CALLSIGN=W1ABC"))

        result = await verify_api_key("valid-api-key", expected_callsign="W1ABC")
        assert result is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_key_with_mismatched_callsign(self):
        """Test verification fails when callsign doesn't match API key owner."""
        respx.post(QRZ_API_URL).mock(return_value=httpx.Response(200, text="RESULT=OK&CALLSIGN=W1AW"))

        result = await verify_api_key("valid-api-key", expected_callsign="W1XYZ")
        assert result is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_key_callsign_case_insensitive(self):
        """Test callsign comparison is case insensitive."""
        respx.post(QRZ_API_URL).mock(return_value=httpx.Response(200, text="RESULT=OK&CALLSIGN=w1abc"))

        result = await verify_api_key("valid-api-key", expected_callsign="W1ABC")
        assert result is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_key_callsign_in_data_field(self):
        """Test callsign extracted from DATA field."""
        respx.post(QRZ_API_URL).mock(return_value=httpx.Response(200, text="RESULT=OK&DATA=TOTAL:100,OWNER:W1ABC,DXCC:50"))

        result = await verify_api_key("valid-api-key", expected_callsign="W1ABC")
        assert result is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_key_no_callsign_in_response_fallback(self):
        """Test key is valid even when no callsign returned (fallback to RESULT=OK)."""
        # If QRZ doesn't return a callsign, we just verify the key works
        respx.post(QRZ_API_URL).mock(return_value=httpx.Response(200, text="RESULT=OK&COUNT=500"))

        result = await verify_api_key("valid-api-key", expected_callsign="W1ABC")
        assert result is True  # Fallback: key works, can't verify ownership

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_key_callsign_suffix_stripped(self):
        """Test /P and /M suffixes are stripped for comparison."""
        respx.post(QRZ_API_URL).mock(return_value=httpx.Response(200, text="RESULT=OK&CALLSIGN=W1ABC"))

        # User enters callsign with suffix, should still match
        result = await verify_api_key("valid-api-key", expected_callsign="W1ABC/P")
        assert result is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_key_api_suffix_stripped(self):
        """Test API returns callsign with suffix, user enters without."""
        respx.post(QRZ_API_URL).mock(return_value=httpx.Response(200, text="RESULT=OK&CALLSIGN=W1ABC/M"))

        # API returns with suffix, user enters without - should match
        result = await verify_api_key("valid-api-key", expected_callsign="W1ABC")
        assert result is True


class TestRealQRZFormat:
    """Test parsing with actual QRZ export format (from KJ5IRF data)."""

    def test_parse_qrz_format_with_sig_info(self):
        """Test parsing actual QRZ export format with SIG_INFO field.

        This format comes from the kj5irf QRZ export file.
        Fields are not in standard order and include QRZ-specific fields.
        """
        # Actual format from QRZ export (reformatted for readability)
        raw = {
            "CALL": "WI0O",
            "QSO_DATE": "20241102",
            "TIME_ON": "1824",
            "BAND": "10m",
            "MODE": "SSB",
            "RST_SENT": "59",
            "RST_RCVD": "59",
            "TX_PWR": "20",
            "GRIDSQUARE": "EN16jc",
            "DXCC": "291",
            "SIG_INFO": "US-8613",  # Park reference from QRZ
            "MY_GRIDSQUARE": "EL09pk",
            "MY_DXCC": "291",
            "QSL_RCVD": "N",
            "APP_QRZLOG_STATUS": "C",  # Confirmed
            "APP_QRZLOG_LOGID": "1234567890",
        }

        result = parse_qso_data(raw)

        assert result is not None
        assert result.dx_callsign == "WI0O"
        assert result.dx_sig_info == "US-8613"  # Park must be preserved
        assert result.dx_grid == "EN16jc"
        assert result.my_grid == "EL09pk"
        assert result.tx_power == 20.0
        assert result.is_confirmed == True  # APP_QRZLOG_STATUS=C means confirmed

    def test_qth_field_not_used_for_park_extraction(self):
        """Test that QTH field is NOT used for park extraction.

        QTH is not a standard ADIF field for POTA park references.
        Park references must be in SIG_INFO (or COMMENT as fallback).
        WRL's non-standard use of QTH for parks is not supported.
        """
        raw = {
            "CALL": "KD8OEY",
            "QSO_DATE": "20241102",
            "TIME_ON": "1835",
            "BAND": "10m",
            "MODE": "SSB",
            "TX_PWR": "20",
            "GRIDSQUARE": "EN71vt",
            "DXCC": "291",
            "QTH": "US-6780",  # Park in QTH field - NOT extracted
            "MY_GRIDSQUARE": "EL09pk",
            "QSL_RCVD": "N",
            "APP_QRZLOG_STATUS": "C",
        }

        result = parse_qso_data(raw)

        assert result is not None
        assert result.dx_callsign == "KD8OEY"
        assert result.dx_sig_info is None  # QTH is not used for park extraction
        assert result.is_confirmed == True

    def test_full_adif_round_trip_with_sig_info(self):
        """Test full ADIF parsing to QSOData with SIG_INFO."""
        # Actual ADIF line format from QRZ export
        adif = """<call:4>WI0O<qso_date:8>20241102<time_on:4>1824<band:3>10m<mode:3>SSB
<tx_pwr:2>20<gridsquare:6>EN16jc<dxcc:3>291<sig_info:7>US-8613
<my_gridsquare:6>EL09pk<app_qrzlog_status:1>C<app_qrzlog_logid:10>1234567890<eor>"""

        # Parse ADIF to raw dict
        records = parse_adif(adif)
        assert len(records) == 1

        # Parse to QSOData
        result = parse_qso_data(records[0])

        assert result is not None
        assert result.dx_callsign == "WI0O"
        assert result.dx_sig_info == "US-8613"
        assert result.dx_grid == "EN16jc"
        assert result.my_grid == "EL09pk"
        assert result.tx_power == 20.0
        assert result.is_confirmed == True
