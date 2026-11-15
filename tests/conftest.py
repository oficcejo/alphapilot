import os
import pytest
from config import Config


@pytest.fixture(autouse=True)
def setup_mock_okx_env(monkeypatch):
    """自动为所有单元测试注入 Mock OKX 凭证，防止缺少环境变量时 is_live() 判定异常。"""
    monkeypatch.setenv("OKX_API_KEY", "mock_test_key")
    monkeypatch.setenv("OKX_API_SECRET", "mock_test_secret")
    monkeypatch.setenv("OKX_API_PASSPHRASE", "mock_test_passphrase")
    monkeypatch.setattr(Config, "OKX_API_KEY", "mock_test_key")
    monkeypatch.setattr(Config, "OKX_API_SECRET", "mock_test_secret")
    monkeypatch.setattr(Config, "OKX_API_PASSPHRASE", "mock_test_passphrase")
