"""
tests/test_okx_client_parsing.py — OKX 客户端响应解析与错误诊断单元测试
"""
import json
import pytest
from unittest.mock import MagicMock
from data_pipeline.okx_client import OKXClient, OKX_ERROR_TRANSLATIONS


def test_parse_response_success():
    client = OKXClient(simulated=True)
    resp = MagicMock()
    resp.json.return_value = {"code": "0", "data": [{"instId": "BTC-USDT-SWAP"}]}
    
    result = client._parse_response(resp, "测试操作")
    assert result["code"] == "0"
    assert result["data"][0]["instId"] == "BTC-USDT-SWAP"


def test_parse_response_50123_trading_permission():
    client = OKXClient(simulated=True)
    resp = MagicMock()
    resp.json.return_value = {
        "code": "50123",
        "msg": "This API Key does not have trading permission for the Crypto",
    }
    
    with pytest.raises(RuntimeError) as exc_info:
        client._parse_response(resp, "OKX 下单")
    
    err_msg = str(exc_info.value)
    assert "50123" in err_msg
    assert "该 API Key 没有此币种的交易权限" in err_msg
    assert "在「交易币种/品种」中勾选「全部」" in err_msg


def test_parse_response_50113_timestamp():
    client = OKXClient(simulated=True)
    resp = MagicMock()
    resp.json.return_value = {
        "code": "50113",
        "msg": "Timestamp request expired",
    }
    
    with pytest.raises(RuntimeError) as exc_info:
        client._parse_response(resp, "OKX 请求")
    
    err_msg = str(exc_info.value)
    assert "50113" in err_msg
    assert "时间戳已过期" in err_msg


def test_parse_response_non_json_fallback():
    client = OKXClient(simulated=True)
    resp = MagicMock()
    resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
    resp.status_code = 502
    resp.text = "Bad Gateway"
    resp.raise_for_status.side_effect = RuntimeError("502 Bad Gateway")
    
    with pytest.raises(RuntimeError):
        client._parse_response(resp, "OKX 请求")
