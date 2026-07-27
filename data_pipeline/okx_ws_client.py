"""
data_pipeline/okx_ws_client.py — OKX v5 WebSocket 实时订阅客户端

功能：
  - 公有频道：订阅 Ticker 行情、K 线实时更新
  - 私有频道：HMAC 认证登录、账户余额、持仓变动、订单状态推送
  - 自动重连 (Auto-reconnect)、心跳维持 (Ping/Pong)
  - 状态回调与实时数据缓存
"""
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Callable, Optional, Dict, Any, List, Set

import aiohttp
from config import Config

logger = logging.getLogger("okx_ws")


def _safe_float(val, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


class OKXWebSocketClient:
    """OKX v5 WebSocket 客户端。"""

    PUBLIC_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
    PRIVATE_WS_URL = "wss://ws.okx.com:8443/ws/v5/private"
    PUBLIC_WS_SIMULATED = "wss://wspap.okx.com:8443/ws/v5/public"
    PRIVATE_WS_SIMULATED = "wss://wspap.okx.com:8443/ws/v5/private"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        simulated: bool = True,
    ):
        self.api_key = api_key or Config.OKX_API_KEY
        self.api_secret = api_secret or Config.OKX_API_SECRET
        self.passphrase = passphrase or Config.OKX_API_PASSPHRASE
        self.simulated = simulated if (api_key or Config.OKX_API_KEY) else True

        # 最新缓存数据
        self.latest_tickers: Dict[str, dict] = {}       # inst_id -> ticker dict
        self.latest_candles: Dict[str, dict] = {}       # inst_id_bar -> candle dict
        self.latest_account: Optional[dict] = None
        self.latest_positions: Dict[str, dict] = {}     # inst_id -> position dict
        self.latest_orders: List[dict] = []

        # 状态控制
        self._is_running = False
        self._connected = False
        self._authenticated = False
        self._last_msg_time = 0.0
        self._msg_count = 0
        self._subscribed_channels: Set[str] = set()

        # 回调订阅
        self._callbacks: List[Callable[[str, Any], None]] = []

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    def get_status(self) -> dict:
        """获取 WebSocket 运行状态。"""
        return {
            "running": self._is_running,
            "connected": self._connected,
            "authenticated": self._authenticated,
            "simulated": self.simulated,
            "last_msg_time": self._last_msg_time,
            "msg_count": self._msg_count,
            "subscribed_channels": list(self._subscribed_channels),
            "cached_tickers": list(self.latest_tickers.keys()),
            "cached_positions": list(self.latest_positions.keys()),
        }

    def add_callback(self, cb: Callable[[str, Any], None]):
        """添加事件回调。"""
        if cb not in self._callbacks:
            self._callbacks.append(cb)

    def _notify(self, event_type: str, data: Any):
        for cb in self._callbacks:
            try:
                cb(event_type, data)
            except Exception as e:
                logger.error(f"WS callback error: {e}")

    # ── 认证签名 ─────────────────────────────────────────────────────────

    def _generate_signature(self, timestamp: str) -> str:
        message = f"{timestamp}GET/users/self/verify"
        mac = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode("utf-8")

    # ── WebSocket 异步监听循环 ─────────────────────────────────────────────

    async def start(self):
        """启动 WebSocket 主循环。"""
        if self._is_running:
            return
        self._is_running = True
        asyncio.create_task(self._public_loop())
        if self.api_key and self.api_secret and self.passphrase:
            asyncio.create_task(self._private_loop())

    def stop(self):
        """停止 WebSocket 主循环。"""
        self._is_running = False
        self._connected = False
        self._authenticated = False

    async def _public_loop(self):
        url = self.PUBLIC_WS_SIMULATED if self.simulated else self.PUBLIC_WS_URL
        backoff = 1

        while self._is_running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, heartbeat=25) as ws:
                        self._connected = True
                        backoff = 1
                        logger.info(f"OKX Public WS Connected: {url}")
                        
                        # 重新订阅通道
                        await self._subscribe_default_public(ws)

                        async for msg in ws:
                            if not self._is_running:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                self._handle_message(msg.data)
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except Exception as e:
                logger.warning(f"OKX Public WS disconnected: {e}. Reconnecting in {backoff}s...")
            finally:
                self._connected = False
                if self._is_running:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)

    async def _private_loop(self):
        url = self.PRIVATE_WS_SIMULATED if self.simulated else self.PRIVATE_WS_URL
        backoff = 1

        while self._is_running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, heartbeat=25) as ws:
                        # 登录认证
                        ts = str(int(time.time()))
                        sign = self._generate_signature(ts)
                        login_msg = {
                            "op": "login",
                            "args": [{
                                "apiKey": self.api_key,
                                "passphrase": self.passphrase,
                                "timestamp": ts,
                                "sign": sign,
                            }]
                        }
                        await ws.send_json(login_msg)
                        logger.info("OKX Private WS Login request sent")

                        async for msg in ws:
                            if not self._is_running:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data_str = msg.data
                                if data_str == "pong":
                                    continue
                                try:
                                    msg_obj = json.loads(data_str)
                                    if msg_obj.get("event") == "login":
                                        if msg_obj.get("code") == "0":
                                            self._authenticated = True
                                            logger.info("OKX Private WS Login Success")
                                            await self._subscribe_private(ws)
                                        else:
                                            logger.error(f"OKX Private WS Login Failed: {msg_obj}")
                                    else:
                                        self._handle_message(data_str)
                                except Exception:
                                    pass
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except Exception as e:
                logger.warning(f"OKX Private WS disconnected: {e}. Reconnecting in {backoff}s...")
            finally:
                self._authenticated = False
                if self._is_running:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)

    async def _subscribe_default_public(self, ws):
        """默认订阅热门 SWAP 品种的 Ticker。"""
        args = [
            {"channel": "tickers", "instId": "BTC-USDT-SWAP"},
            {"channel": "tickers", "instId": "ETH-USDT-SWAP"},
        ]
        sub_msg = {"op": "subscribe", "args": args}
        await ws.send_json(sub_msg)
        for a in args:
            self._subscribed_channels.add(f"{a['channel']}:{a['instId']}")

    async def _subscribe_private(self, ws):
        """订阅私有持仓和账户。"""
        args = [
            {"channel": "account"},
            {"channel": "positions", "instType": "SWAP"},
            {"channel": "orders", "instType": "SWAP"},
        ]
        sub_msg = {"op": "subscribe", "args": args}
        await ws.send_json(sub_msg)
        for a in args:
            self._subscribed_channels.add(f"private:{a['channel']}")

    def subscribe_symbol_ticker(self, inst_id: str):
        """记录准备订阅的品种。"""
        self._subscribed_channels.add(f"tickers:{inst_id}")

    # ── 消息处理 ──────────────────────────────────────────────────────────

    def _handle_message(self, raw_str: str):
        if raw_str == "pong":
            return
        self._last_msg_time = time.time()
        self._msg_count += 1

        try:
            msg = json.loads(raw_str)
        except Exception:
            return

        arg = msg.get("arg", {})
        channel = arg.get("channel", "")
        data = msg.get("data", [])

        if channel == "tickers" and data:
            for item in data:
                inst_id = item.get("instId")
                if inst_id:
                    self.latest_tickers[inst_id] = {
                        "inst_id": inst_id,
                        "last": _safe_float(item.get("last")),
                        "ask": _safe_float(item.get("askPx")),
                        "bid": _safe_float(item.get("bidPx")),
                        "high_24h": _safe_float(item.get("high24h")),
                        "low_24h": _safe_float(item.get("low24h")),
                        "vol_24h": _safe_float(item.get("vol24h")),
                        "timestamp": int(item.get("ts", time.time() * 1000)),
                    }
            self._notify("ticker", self.latest_tickers)

        elif channel == "positions" and data:
            for item in data:
                inst_id = item.get("instId")
                if inst_id:
                    self.latest_positions[inst_id] = {
                        "inst_id": inst_id,
                        "pos_side": item.get("posSide", "net"),
                        "pos": _safe_float(item.get("pos")),
                        "avg_px": _safe_float(item.get("avgPx")),
                        "upl": _safe_float(item.get("upl")),
                        "lever": _safe_float(item.get("lever")),
                    }
            self._notify("positions", self.latest_positions)

        elif channel == "account" and data:
            item = data[0]
            self.latest_account = {
                "total_eq": _safe_float(item.get("totalEq")),
                "upl": _safe_float(item.get("upl")),
                "avail_bal": _safe_float(item.get("availBal")),
            }
            self._notify("account", self.latest_account)

        elif channel == "orders" and data:
            self.latest_orders.extend(data)
            # 保留最近 100 笔
            self.latest_orders = self.latest_orders[-100:]
            self._notify("orders", data)


# 全局单例
okx_ws_client = OKXWebSocketClient()
