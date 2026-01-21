"""
Tests for LoTW (Logbook of the World) client.
"""

import pytest
import respx
import httpx
from datetime import datetime

from lotw_client import (
    verify_lotw_credentials,
    fetch_lotw_qsos,
    parse_lotw_qso,
    LoTWError,
    LOTW_REPORT_URL,
)


class TestParseLotWQSO:
    """Test converting raw ADIF to QSOData objects for LoTW."""

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

        result = parse_lotw_qso(raw)

        assert result is not None
        assert result.dx_callsign == "DL1ABC"
        assert result.qso_datetime == datetime(2026, 1, 1, 12, 1, 0)
        assert result.band == "20M"
        assert result.mode == "SSB"
        assert result.tx_power == 5.0
        assert result.my_grid == "EM12"
        assert result.dx_grid == "JN58"
        assert result.dx_dxcc == 230
        assert result.is_confirmed is True

    def test_parse_lotw_qsl_confirmed(self):
        """Test parsing QSO confirmed via LoTW QSL field."""
        raw = {
            "CALL": "K1ABC",
            "QSO_DATE": "20260115",
            "TIME_ON": "0830",
            "LOTW_QSL_RCVD": "Y",
        }

        result = parse_lotw_qso(raw)
        assert result.is_confirmed is True

    def test_parse_missing_call(self):
        """Test parsing QSO with missing CALL returns None."""
        raw = {
            "QSO_DATE": "20260101",
            "TIME_ON": "1200",
        }

        result = parse_lotw_qso(raw)
        assert result is None

    def test_parse_missing_date(self):
        """Test parsing QSO with missing date returns None."""
        raw = {
            "CALL": "W1ABC",
            "TIME_ON": "1200",
        }

        result = parse_lotw_qso(raw)
        assert result is None

    def test_parse_six_digit_time(self):
        """Test parsing QSO with HHMMSS time format."""
        raw = {
            "CALL": "K3ABC",
            "QSO_DATE": "20260120",
            "TIME_ON": "143025",
        }

        result = parse_lotw_qso(raw)
        assert result.qso_datetime == datetime(2026, 1, 20, 14, 30, 25)

    def test_parse_four_digit_time(self):
        """Test parsing QSO with HHMM time format (padded to 6 digits)."""
        raw = {
            "CALL": "K3ABC",
            "QSO_DATE": "20260120",
            "TIME_ON": "1430",
        }

        result = parse_lotw_qso(raw)
        assert result.qso_datetime == datetime(2026, 1, 20, 14, 30, 0)

    def test_parse_default_time(self):
        """Test parsing QSO with missing time defaults to 0000."""
        raw = {
            "CALL": "K3ABC",
            "QSO_DATE": "20260120",
        }

        result = parse_lotw_qso(raw)
        assert result.qso_datetime == datetime(2026, 1, 20, 0, 0, 0)

    def test_parse_invalid_datetime(self):
        """Test parsing QSO with invalid datetime returns None."""
        raw = {
            "CALL": "W1ABC",
            "QSO_DATE": "invalid",
            "TIME_ON": "1200",
        }
        result = parse_lotw_qso(raw)
        assert result is None

    def test_parse_invalid_tx_power(self):
        """Test parsing QSO with invalid TX power."""
        raw = {
            "CALL": "W1ABC",
            "QSO_DATE": "20260101",
            "TIME_ON": "1200",
            "TX_PWR": "not_a_number",
        }
        result = parse_lotw_qso(raw)
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
        result = parse_lotw_qso(raw)
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
        result = parse_lotw_qso(raw)
        assert result is not None
        assert result.dx_dxcc is None

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
        }

        result = parse_lotw_qso(raw)

        assert result.my_grid == "DN15"
        assert result.my_sig_info == "K-0001"
        assert result.my_dxcc == 291
        assert result.dx_grid == "CM87"

    def test_qrz_logid_is_none(self):
        """Test that LoTW QSOs have no QRZ log ID."""
        raw = {
            "CALL": "W1ABC",
            "QSO_DATE": "20260101",
            "TIME_ON": "1200",
        }
        result = parse_lotw_qso(raw)
        assert result.qrz_logid is None

    def test_callsign_uppercase(self):
        """Test that callsign is converted to uppercase."""
        raw = {
            "CALL": "w1abc",
            "QSO_DATE": "20260101",
            "TIME_ON": "1200",
        }
        result = parse_lotw_qso(raw)
        assert result.dx_callsign == "W1ABC"

    def test_qth_field_not_used_for_park_extraction(self):
        """Test that QTH field is NOT used for park extraction.

        QTH is not a standard ADIF field for POTA park references.
        Park references must be in SIG_INFO (or COMMENT as fallback).
        """
        raw = {
            "CALL": "W1ABC",
            "QSO_DATE": "20260115",
            "TIME_ON": "1423",
            "QTH": "US-8613",  # Park in QTH - NOT extracted
        }

        result = parse_lotw_qso(raw)

        assert result is not None
        assert result.dx_sig_info is None  # QTH is not used for park extraction

    def test_my_qth_field_not_used_for_park_extraction(self):
        """Test that MY_QTH field is NOT used for park extraction."""
        raw = {
            "CALL": "W1ABC",
            "QSO_DATE": "20260115",
            "TIME_ON": "1423",
            "MY_QTH": "K-4556",  # Park in MY_QTH - NOT extracted
        }

        result = parse_lotw_qso(raw)

        assert result is not None
        assert result.my_sig_info is None  # MY_QTH is not used for park extraction


class TestVerifyLoTWCredentials:
    """Test LoTW credential verification."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_valid_credentials(self):
        """Test verification of valid credentials."""
        respx.get(LOTW_REPORT_URL).mock(
            return_value=httpx.Response(200, text="ARRL Logbook of the World\n<PROGRAMID:4>LoTW\n<EOH>")
        )

        result = await verify_lotw_credentials("W1ABC", "password123")
        assert result is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_valid_with_eoh(self):
        """Test verification with EOH marker."""
        respx.get(LOTW_REPORT_URL).mock(
            return_value=httpx.Response(200, text="<EOH>\n")
        )

        result = await verify_lotw_credentials("W1ABC", "password123")
        assert result is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_invalid_password(self):
        """Test verification with invalid password."""
        respx.get(LOTW_REPORT_URL).mock(
            return_value=httpx.Response(200, text="Username/password incorrect")
        )

        result = await verify_lotw_credentials("W1ABC", "wrongpassword")
        assert result is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_account_not_found(self):
        """Test verification when account not found."""
        respx.get(LOTW_REPORT_URL).mock(
            return_value=httpx.Response(200, text="User not found in LoTW database")
        )

        result = await verify_lotw_credentials("INVALID", "password")
        assert result is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_http_error(self):
        """Test verification when HTTP error occurs."""
        respx.get(LOTW_REPORT_URL).mock(side_effect=httpx.ConnectError("Connection failed"))

        result = await verify_lotw_credentials("W1ABC", "password")
        assert result is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_with_expected_callsign_match(self):
        """Test verification passes when callsign matches username."""
        respx.get(LOTW_REPORT_URL).mock(
            return_value=httpx.Response(200, text="<PROGRAMID:4>LoTW\n<EOH>")
        )

        result = await verify_lotw_credentials("W1ABC", "password", expected_callsign="W1ABC")
        assert result is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_with_expected_callsign_mismatch(self):
        """Test verification fails when callsign doesn't match username."""
        respx.get(LOTW_REPORT_URL).mock(
            return_value=httpx.Response(200, text="<PROGRAMID:4>LoTW\n<EOH>")
        )

        result = await verify_lotw_credentials("W1ABC", "password", expected_callsign="K2XYZ")
        assert result is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_callsign_case_insensitive(self):
        """Test callsign comparison is case insensitive."""
        respx.get(LOTW_REPORT_URL).mock(
            return_value=httpx.Response(200, text="<PROGRAMID:4>LoTW\n<EOH>")
        )

        result = await verify_lotw_credentials("w1abc", "password", expected_callsign="W1ABC")
        assert result is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_verify_unexpected_response(self):
        """Test verification fails on unexpected response."""
        respx.get(LOTW_REPORT_URL).mock(
            return_value=httpx.Response(200, text="Some unexpected content")
        )

        result = await verify_lotw_credentials("W1ABC", "password")
        assert result is False


class TestFetchLoTWQSOs:
    """Test async LoTW API fetching."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_success(self):
        """Test successful QSO fetch."""
        adif_response = """ARRL Logbook of the World
<PROGRAMID:4>LoTW
<EOH>
<CALL:5>DL1AB<QSO_DATE:8>20260115<TIME_ON:4>1423<BAND:3>20M<MODE:3>SSB<QSL_RCVD:1>Y<EOR>
"""
        respx.get(LOTW_REPORT_URL).mock(return_value=httpx.Response(200, text=adif_response))

        result = await fetch_lotw_qsos("W1ABC", "password123")

        assert len(result) == 1
        assert result[0].dx_callsign == "DL1AB"

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_multiple(self):
        """Test fetching multiple QSOs."""
        adif_response = """<EOH>
<CALL:5>DL1AB<QSO_DATE:8>20260115<TIME_ON:4>1423<EOR>
<CALL:5>JA1XY<QSO_DATE:8>20260116<TIME_ON:4>0800<EOR>
"""
        respx.get(LOTW_REPORT_URL).mock(return_value=httpx.Response(200, text=adif_response))

        result = await fetch_lotw_qsos("W1ABC", "password123")

        assert len(result) == 2
        assert result[0].dx_callsign == "DL1AB"
        assert result[1].dx_callsign == "JA1XY"

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_auth_failure(self):
        """Test authentication failure."""
        respx.get(LOTW_REPORT_URL).mock(
            return_value=httpx.Response(200, text="Username/password incorrect")
        )

        with pytest.raises(LoTWError) as exc:
            await fetch_lotw_qsos("W1ABC", "wrongpass")

        assert "Authentication failed" in str(exc.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_account_not_found(self):
        """Test account not found error."""
        respx.get(LOTW_REPORT_URL).mock(
            return_value=httpx.Response(200, text="User not found in LoTW")
        )

        with pytest.raises(LoTWError) as exc:
            await fetch_lotw_qsos("INVALID", "password")

        assert "not found" in str(exc.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_http_error(self):
        """Test HTTP error handling."""
        respx.get(LOTW_REPORT_URL).mock(side_effect=httpx.ConnectError("Connection failed"))

        with pytest.raises(LoTWError) as exc:
            await fetch_lotw_qsos("W1ABC", "password")

        assert "HTTP error" in str(exc.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_empty_logbook(self):
        """Test empty logbook returns empty list."""
        respx.get(LOTW_REPORT_URL).mock(
            return_value=httpx.Response(200, text="<PROGRAMID:4>LoTW\n<EOH>\n")
        )

        result = await fetch_lotw_qsos("W1ABC", "password")
        assert result == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_confirmed_only(self):
        """Test confirmed_only parameter is passed."""
        respx.get(LOTW_REPORT_URL).mock(return_value=httpx.Response(200, text="<EOH>"))

        await fetch_lotw_qsos("W1ABC", "password", confirmed_only=True)

        # Check that qso_qsl=yes was in the request
        assert respx.calls[0].request.url.params.get("qso_qsl") == "yes"

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_all_qsos(self):
        """Test fetching all QSOs (not just confirmed)."""
        respx.get(LOTW_REPORT_URL).mock(return_value=httpx.Response(200, text="<EOH>"))

        await fetch_lotw_qsos("W1ABC", "password", confirmed_only=False)

        # Check that qso_qsl was not in the request
        assert "qso_qsl" not in respx.calls[0].request.url.params

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_with_start_date(self):
        """Test start_date parameter is passed."""
        respx.get(LOTW_REPORT_URL).mock(return_value=httpx.Response(200, text="<EOH>"))

        await fetch_lotw_qsos("W1ABC", "password", start_date="2026-01-01")

        assert respx.calls[0].request.url.params.get("qso_startdate") == "2026-01-01"

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_qsos_skips_invalid(self):
        """Test that invalid QSOs are skipped."""
        # One valid QSO, one missing CALL
        adif_response = """<EOH>
<CALL:5>W1ABC<QSO_DATE:8>20260115<TIME_ON:4>1400<EOR>
<QSO_DATE:8>20260115<TIME_ON:4>1500<EOR>
"""
        respx.get(LOTW_REPORT_URL).mock(return_value=httpx.Response(200, text=adif_response))

        result = await fetch_lotw_qsos("W1ABC", "password")

        assert len(result) == 1
        assert result[0].dx_callsign == "W1ABC"


class TestSensitiveDataFilter:
    """Test that sensitive data is redacted from logs."""

    def test_password_redacted_from_log_message(self):
        """Test password query parameter is redacted."""
        import logging
        from main import SensitiveDataFilter

        filter = SensitiveDataFilter()
        record = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='HTTP Request: GET https://lotw.arrl.org/lotwreport.adi?login=W1ABC&password=secret123&qso_query=1 "HTTP/1.1 200 OK"',
            args=(),
            exc_info=None,
        )

        filter.filter(record)

        assert "secret123" not in record.msg
        assert "password=[REDACTED]" in record.msg
        assert "login=W1ABC" in record.msg  # Non-sensitive params preserved

    def test_api_key_redacted(self):
        """Test api_key query parameter is redacted."""
        import logging
        from main import SensitiveDataFilter

        filter = SensitiveDataFilter()
        record = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='HTTP Request: GET https://api.example.com?api_key=abc123xyz&data=test "HTTP/1.1 200 OK"',
            args=(),
            exc_info=None,
        )

        filter.filter(record)

        assert "abc123xyz" not in record.msg
        assert "api_key=[REDACTED]" in record.msg

    def test_key_param_redacted(self):
        """Test key query parameter is redacted."""
        import logging
        from main import SensitiveDataFilter

        filter = SensitiveDataFilter()
        record = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='HTTP Request: POST https://logbook.qrz.com/api?key=myapikey123 "HTTP/1.1 200 OK"',
            args=(),
            exc_info=None,
        )

        filter.filter(record)

        assert "myapikey123" not in record.msg
        assert "key=[REDACTED]" in record.msg

    def test_case_insensitive_redaction(self):
        """Test redaction is case insensitive."""
        import logging
        from main import SensitiveDataFilter

        filter = SensitiveDataFilter()
        record = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='GET https://example.com?PASSWORD=Secret123&API_KEY=key456',
            args=(),
            exc_info=None,
        )

        filter.filter(record)

        assert "Secret123" not in record.msg
        assert "key456" not in record.msg

    def test_multiple_sensitive_params(self):
        """Test multiple sensitive params in same URL are all redacted."""
        import logging
        from main import SensitiveDataFilter

        filter = SensitiveDataFilter()
        record = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='GET https://example.com?password=pass1&api_key=key2&key=key3',
            args=(),
            exc_info=None,
        )

        filter.filter(record)

        assert "pass1" not in record.msg
        assert "key2" not in record.msg
        assert "key3" not in record.msg
        assert record.msg.count("[REDACTED]") == 3

    def test_non_sensitive_message_unchanged(self):
        """Test non-sensitive messages pass through unchanged."""
        import logging
        from main import SensitiveDataFilter

        filter = SensitiveDataFilter()
        original_msg = 'HTTP Request: GET https://example.com/data?page=1&limit=10 "HTTP/1.1 200 OK"'
        record = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=original_msg,
            args=(),
            exc_info=None,
        )

        filter.filter(record)

        assert record.msg == original_msg
