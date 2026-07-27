"""
data_pipeline/okx_client.py — OKX v5 REST API 客户端

功能：
  - 公有接口：品种发现、K线下载、行情
  - 私有接口：账户余额、持仓、下单

合规声明：
  - 默认模拟盘（x-simulated-trading: 1）
  - 实盘需显式设置 TRADING_MODE=live + 完整凭证
"""
import hmac
import base64
import hashlib
import time
import json
from typing import Optional

import requests

from config import Config

BROKER_TAG = "c314b0aecb5bBCDE"


def _safe_float(val, default: float = 0.0) -> float:
    """安全转换为 float，处理 OKX API 返回的空字符串 ''。"""
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


class OKXClient:
    """OKX v5 REST API 客户端。"""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        base_url: str = "",
        simulated: bool = True,
    ):
        self.api_key = api_key or Config.OKX_API_KEY
        self.api_secret = api_secret or Config.OKX_API_SECRET
        self.passphrase = passphrase or Config.OKX_API_PASSPHRASE
        self.base_url = (base_url or Config.OKX_API_BASE).rstrip("/")
        self.simulated = simulated if (api_key or Config.OKX_API_KEY) else True

        self.broker_tag = BROKER_TAG
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
        })

    # ── 认证 ──────────────────────────────────────────────────────────────

    def _timestamp(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        msg = f"{timestamp}{method.upper()}{path}{body}"
        mac = hmac.new(
            self.api_secret.encode("utf-8"),
            msg.encode("utf-8"),
            hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode("utf-8")

    def _auth_headers(self, method: str, path: str, body: str = "") -> dict:
        ts = self._timestamp()
        return {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": self._sign(ts, method, path, body),
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "x-simulated-trading": "1" if self.simulated else "0",
        }

    # ── 公有接口 ──────────────────────────────────────────────────────────

    def get_instruments(self, inst_type: str = "SWAP") -> list[dict]:
        """获取可用合约品种列表。

        Args:
            inst_type: SWAP / FUTURES / SPOT / OPTION
        """
        path = f"/api/v5/public/instruments?instType={inst_type}"
        url = self.base_url + path
        resp = self._session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX API error: {data.get('msg', data)}")
        return data.get("data", [])

    def get_instrument(self, inst_id: str, inst_type: str = "SWAP") -> dict:
        """查询单个合约信息（含 ctVal 合约面值、ctValCcy 面值币种、lotSz 下单步进、minSz 最小下单量）。

        Args:
            inst_id: 合约 ID，如 BTC-USDT-SWAP
            inst_type: SWAP / FUTURES / SPOT / OPTION

        Returns:
            合约信息字典，关键字段：
              - ctVal: 每张合约面值（如 0.01 表示 1 张 = 0.01 BTC）
              - ctValCcy: 面值币种（如 "BTC"）
              - lotSz: 下单数量步进
              - minSz: 最小下单数量
              - ctMult: 合约乘数
              - tickSz: 价格步进
              - settleCcy: 结算币种
              - quoteCcy: 计价币种
        """
        path = f"/api/v5/public/instruments?instType={inst_type}&instId={inst_id}"
        url = self.base_url + path
        resp = self._session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX API error: {data.get('msg', data)}")
        items = data.get("data", [])
        return items[0] if items else {}

    def get_candles(
        self,
        inst_id: str,
        bar: str = "1H",
        limit: int = 300,
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> list[list]:
        """获取 K 线数据（公有接口，无需认证）。

        Args:
            inst_id: 如 BTC-USDT-SWAP
            bar: K线周期 1m/5m/15m/30m/1H/2H/4H/6H/12H/1D/1W/1M
            limit: 最多 300 根
            after: 分页游标（返回此时间戳之前的数据）
            before: 分页游标

        Returns:
            OKX K线列表 [[ts, o, h, l, c, vol, ...], ...]
        """
        params = f"instId={inst_id}&bar={bar}&limit={limit}"
        if after:
            params += f"&after={after}"
        if before:
            params += f"&before={before}"
        path = f"/api/v5/market/candles?{params}"
        url = self.base_url + path
        resp = self._session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX API error: {data.get('msg', data)}")
        return data.get("data", [])

    def get_candles_history(
        self,
        inst_id: str,
        bar: str = "1H",
        limit: int = 100,
        after: Optional[str] = None,
    ) -> list[list]:
        """获取历史 K 线（支持更早的数据）。"""
        params = f"instId={inst_id}&bar={bar}&limit={limit}"
        if after:
            params += f"&after={after}"
        path = f"/api/v5/market/history-candles?{params}"
        url = self.base_url + path
        resp = self._session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX API error: {data.get('msg', data)}")
        return data.get("data", [])

    def get_ticker(self, inst_id: str) -> dict:
        """获取最新行情。"""
        path = f"/api/v5/market/ticker?instId={inst_id}"
        url = self.base_url + path
        resp = self._session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX API error: {data.get('msg', data)}")
        items = data.get("data", [])
        return items[0] if items else {}

    def download_candles(
        self,
        inst_id: str,
        bar: str = "1H",
        total_bars: int = 2000,
    ) -> list[list]:
        """分页下载大量 K 线数据。

        OKX 每次最多返回 300 根，使用 after 游标向前翻页。
        """
        all_candles: list[list] = []
        after = None
        remaining = total_bars

        while remaining > 0:
            batch_limit = min(100, remaining)
            try:
                batch = self.get_candles_history(inst_id, bar, batch_limit, after)
            except Exception:
                # history-candles 失败则用普通 candles
                batch = self.get_candles(inst_id, bar, batch_limit, after)

            if not batch:
                break

            all_candles.extend(batch)
            # OKX 返回按时间倒序，最后一条是最早的
            after = batch[-1][0]
            remaining = total_bars - len(all_candles)

            # 避免 API 限流
            time.sleep(0.15)

            if len(batch) < batch_limit:
                break

        # 去重 + 按时间正序
        seen = set()
        unique = []
        for c in all_candles:
            ts = c[0]
            if ts not in seen:
                seen.add(ts)
                unique.append(c)
        unique.sort(key=lambda x: int(x[0]))
        return unique

    # ── 私有接口 ──────────────────────────────────────────────────────────

    def get_account_balance(self) -> dict:
        """获取账户余额。"""
        path = "/api/v5/account/balance"
        body = ""
        headers = self._auth_headers("GET", path, body)
        url = self.base_url + path
        resp = self._session.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX API error: {data.get('msg', data)}")
        items = data.get("data", [])
        return items[0] if items else {}

    def get_account_summary(self) -> dict:
        """提取账户余额摘要（结构化）。

        Returns:
            {
                "total_eq": 总权益 (USDT),
                "upl": 未实现盈亏,
                "upl_ratio": 未实现盈亏率,
                "margin": 保证金占用,
                "margin_ratio": 保证金率,
                "ord_froz": 委托冻结,
                "avail_bal": 可用余额,
                "currency": 币种,
                "details": [{币种, 权益, 余额, 盈亏}, ...]
            }
        """
        raw = self.get_account_balance()
        if not raw:
            return {
                "total_eq": 0.0, "upl": 0.0, "upl_ratio": 0.0,
                "margin": 0.0, "margin_ratio": 0.0, "ord_froz": 0.0,
                "avail_bal": 0.0, "currency": "USDT", "details": [],
            }
        # OKX API 对无持仓账户会返回空字符串 ""，需用 _safe_float 处理
        details = []
        for d in raw.get("details", []):
            details.append({
                "ccy": d.get("ccy", ""),
                "eq": _safe_float(d.get("eq")),
                "avail_bal": _safe_float(d.get("availBal")),
                "cash_bal": _safe_float(d.get("cashBal")),
                "upl": _safe_float(d.get("upl")),
            })
        avail_bal = next((d["avail_bal"] for d in details if d["ccy"] == "USDT"), 0.0)
        return {
            "total_eq": _safe_float(raw.get("totalEq")),
            "upl": _safe_float(raw.get("upl")),
            "upl_ratio": _safe_float(raw.get("uplRatio")),
            "margin": _safe_float(raw.get("margin")),
            "margin_ratio": _safe_float(raw.get("mgnRatio")),
            "ord_froz": _safe_float(raw.get("ordFroz")),
            "avail_bal": avail_bal,
            "currency": "USDT",
            "details": details,
        }

    def get_positions(self, inst_id: Optional[str] = None) -> list[dict]:
        """获取持仓信息。"""
        path = "/api/v5/account/positions"
        if inst_id:
            path += f"?instId={inst_id}"
        headers = self._auth_headers("GET", path, "")
        url = self.base_url + path
        resp = self._session.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX API error: {data.get('msg', data)}")
        return data.get("data", [])

    def get_positions_detail(self, inst_id: Optional[str] = None) -> list[dict]:
        """提取持仓详情（结构化）。

        Returns:
            [{
                "inst_id": 合约ID,
                "pos_side": 持仓方向 long/short/net,
                "pos": 持仓数量（张数）,
                "pos_ccy": 持仓币种,
                "avg_px": 开仓均价,
                "last": 最新价,
                "upl": 未实现盈亏,
                "upl_ratio": 未实现盈亏率,
                "realized_pnl": 已实现盈亏,
                "margin": 保证金,
                "mgn_mode": 保证金模式 cross/isolated,
                "lever": 杠杆,
                "liq_px": 强平价,
            }, ...]
        """
        raw_list = self.get_positions(inst_id)
        result = []
        for p in raw_list:
            try:
                result.append({
                    "inst_id": p.get("instId", ""),
                    "pos_side": p.get("posSide", "net"),
                    "pos": _safe_float(p.get("pos")),
                    "pos_ccy": p.get("posCcy", ""),
                    "avg_px": _safe_float(p.get("avgPx")),
                    "last": _safe_float(p.get("last")),
                    "upl": _safe_float(p.get("upl")),
                    "upl_ratio": _safe_float(p.get("uplRatio")),
                    "realized_pnl": _safe_float(p.get("realizedPnl")),
                    "margin": _safe_float(p.get("margin")),
                    "mgn_mode": p.get("mgnMode", "cross"),
                    "lever": _safe_float(p.get("lever")),
                    "liq_px": _safe_float(p.get("liqPx")),
                })
            except Exception:
                continue
        return result

    def place_order(
        self,
        inst_id: str,
        side: str,          # buy / sell
        pos_side: str = "net",  # long / short / net
        ord_type: str = "market",  # market / limit
        sz: str = "",       # 数量
        px: str = "",       # 限价价格
        td_mode: str = "cross",  # cross / isolated / cash
        cl_ord_id: str = "",
        tag: Optional[str] = None,
    ) -> dict:
        """下单。

        安全闸门：仅当 Config.is_live() 为 True 时才发送真实订单，
        否则返回模拟确认。
        """
        broker_tag = tag if tag is not None else BROKER_TAG

        # ── 安全闸门：非实盘模式返回模拟确认 ──
        if not Config.is_live():
            return {
                "simulated": True,
                "mode": Config.TRADING_MODE,
                "inst_id": inst_id,
                "side": side,
                "pos_side": pos_side,
                "ord_type": ord_type,
                "sz": sz,
                "px": px,
                "td_mode": td_mode,
                "cl_ord_id": cl_ord_id,
                "broker_tag": broker_tag,
                "msg": "PAPER / SIMULATED — 未发送真实订单。设置 TRADING_MODE=live + 完整凭证以启用实盘。",
                "timestamp": int(time.time() * 1000),
            }

        # ── 实盘下单 ──
        body_dict = {
            "instId": inst_id,
            "tdMode": td_mode,
            "side": side,
            "posSide": pos_side,
            "ordType": ord_type,
            "sz": str(sz),
        }
        if ord_type == "limit" and px:
            body_dict["px"] = str(px)
        if cl_ord_id:
            body_dict["clOrdId"] = cl_ord_id
        body_dict["tag"] = broker_tag

        body = json.dumps(body_dict)
        path = "/api/v5/trade/order"
        headers = self._auth_headers("POST", path, body)
        url = self.base_url + path
        resp = self._session.post(url, headers=headers, data=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            # OKX 返回两层错误：顶层 msg + data[0].sMsg（具体原因）
            top_msg = data.get("msg", "")
            detail_msgs = []
            for d in data.get("data", []):
                s_code = d.get("sCode", "")
                s_msg = d.get("sMsg", "")
                if s_code and s_code != "0":
                    detail_msgs.append(f"[{s_code}] {s_msg}")
            detail_str = "; ".join(detail_msgs) if detail_msgs else ""
            raise RuntimeError(
                f"OKX order error: {top_msg}"
                + (f" — {detail_str}" if detail_str else "")
                + f" | instId={inst_id} sz={sz} side={side} tdMode={td_mode} posSide={pos_side}"
            )
        result = data.get("data", [{}])[0]
        # 检查单笔订单是否成功（sCode != "0" 表示该笔失败）
        if result.get("sCode") and result.get("sCode") != "0":
            raise RuntimeError(
                f"OKX order rejected: [{result.get('sCode')}] {result.get('sMsg')}"
                + f" | instId={inst_id} sz={sz} side={side} tdMode={td_mode} posSide={pos_side}"
            )
        result["broker_tag"] = broker_tag
        result["live"] = True
        return result

    def close_position(self, inst_id: str, pos_side: str = "net", mgn_mode: str = "cross") -> dict:
        """平仓。"""
        if not Config.is_live():
            return {
                "simulated": True,
                "inst_id": inst_id,
                "pos_side": pos_side,
                "msg": "PAPER — 模拟平仓",
            }
        body_dict = {"instId": inst_id, "mgnMode": mgn_mode}
        if pos_side != "net":
            body_dict["posSide"] = pos_side
        body = json.dumps(body_dict)
        path = "/api/v5/trade/close-position"
        headers = self._auth_headers("POST", path, body)
        url = self.base_url + path
        resp = self._session.post(url, headers=headers, data=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX close error: {data.get('msg', data)}")
        return data.get("data", [{}])[0]

    def get_account_config(self) -> dict:
        """获取账户配置信息。

        关键字段：
          - posMode: 持仓模式
            "long_short_mode" = 双向持仓（posSide 传 long/short）
            "net_mode"        = 单向持仓（posSide 传 net）
        """
        if not Config.is_live():
            return {"simulated": True, "posMode": "net_mode", "msg": "PAPER — 模拟账户配置"}
        path = "/api/v5/account/config"
        headers = self._auth_headers("GET", path, "")
        url = self.base_url + path
        resp = self._session.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"OKX account config error: {data.get('msg', data)}")
        items = data.get("data", [])
        return items[0] if items else {}

    def set_leverage(self, inst_id: str, lever: int, mgn_mode: str = "cross", pos_side: str = "") -> dict:
        """设置杠杆。"""
        if not Config.is_live():
            return {"simulated": True, "inst_id": inst_id, "lever": lever, "msg": "PAPER — 模拟设置杠杆"}
        body_dict = {"instId": inst_id, "lever": str(lever), "mgnMode": mgn_mode}
        if pos_side:
            body_dict["posSide"] = pos_side
        body = json.dumps(body_dict)
        path = "/api/v5/account/set-leverage"
        headers = self._auth_headers("POST", path, body)
        url = self.base_url + path
        resp = self._session.post(url, headers=headers, data=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data


# ── 便捷工厂 ───────────────────────────────────────────────────────────────

def get_public_client() -> OKXClient:
    """获取公有接口客户端（无需认证）。"""
    return OKXClient(simulated=True)


def get_private_client() -> OKXClient:
    """获取私有接口客户端（需认证）。"""
    return OKXClient(
        simulated=Config.OKX_API_SIMULATED,
    )
