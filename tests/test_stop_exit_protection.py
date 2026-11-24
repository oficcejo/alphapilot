"""
tests/test_stop_exit_protection.py — 止损与仓位执行保护 (2026-11 实盘复盘后的加固)

覆盖：
  1. 止损锚点：交易所端止损单按下单瞬间的实时价锚定 (WS → REST → 收盘价兜底)，
     不再按上一根已收盘 K 线的收盘价；实际成交均价写入审计
  2. 止损冷却：交易所端止损 / 保本 / 追踪止损出场后，冷却期内禁止同向开仓与加仓；
     减仓、平仓、反向不受影响；已过期、非本机器人、关闭开关、查询失败均不拦截；
     并用 2026-11 实盘中真实发生的 8 次"止损后重进"时间回放默认值的效果
  3. 仓位规模：按 min(本金, 实时权益) 计算，只缩不放
"""
import datetime as dt
import math
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from config import Config
from api.services.trading_service import (
    TradingService,
    clamp_target_for_stop_cooldown,
    compute_effective_capital,
    stop_cooldown_until,
)

INST = "BTC-USDT-SWAP"
# BTC-USDT-SWAP 口径: ctVal=0.01 BTC, lotSz=minSz=0.01 张, tickSz=0.1
BTC_INST = {"ctVal": "0.01", "lotSz": "0.01", "minSz": "0.01", "tickSz": "0.1", "ctValCcy": "BTC"}
BAR_SEC = 1800  # 测试统一用 30m


@pytest.fixture
def service():
    svc = TradingService()
    svc._instrument_cache[INST] = dict(BTC_INST)
    return svc


# ── 测试辅助 ──────────────────────────────────────────────────────────────

def _candles(px=64000.0):
    """800 根 30m K 线，全部收在 px。"""
    return [
        [str(1600000000000 + i * BAR_SEC * 1000), str(px), str(px + 100), str(px - 100), str(px), "100", "", "", "1"]
        for i in range(800)
    ]


def _ts_in_current_bar(frac=0.5):
    """当前这根 K 线内、已经过去的某个时刻 (默认已过部分的中间点)。"""
    now = time.time()
    bar_start = math.floor(now / BAR_SEC) * BAR_SEC
    return bar_start + (now - bar_start) * frac



def _stopped_algo(side, trigger_ts, algo_id="sl1795445372533"):
    """一张本机器人挂出、在 trigger_ts 触发的止损单。side 是平仓方向 (sell=平多)。"""
    return {
        "algoClOrdId": algo_id,
        "side": side,
        "sz": "1.37",
        "actualSz": "1.37",
        "slTriggerPx": "61671.2",
        "triggerTime": str(int(trigger_ts * 1000)),
    }


def _run(service, client, signal, current_sz=0.0, capital=10000.0, eq=10000.0, upl=0.0, price=64000.0,
         ws_last=None, ws_fresh=True, history=None):
    """以实盘模式跑一次 execute_signal，client 为 MagicMock。返回 (结果, ws 桩)。

    history: 交易所端"已触发的止损单"列表，默认为空。
    ws_fresh: WebSocket 是否在线且有新消息 (默认是)。
    """
    client.get_positions_detail.return_value = (
        [{"inst_id": INST, "pos_side": "net", "pos": current_sz, "avg_px": price}] if current_sz else []
    )
    client.get_account_summary.return_value = {"total_eq": eq, "avail_bal": eq, "upl": upl}
    client.get_recent_candles.return_value = _candles(price)
    client.get_algo_orders_history.return_value = history if history is not None else []
    client.place_order.return_value = {"clOrdId": "ap1", "sCode": "0", "live": True}
    client.get_order.return_value = {"state": "filled", "avgPx": "64000.5", "accFillSz": "0.50"}

    ws = MagicMock()
    ws.get_status.return_value = {"connected": ws_fresh, "last_msg_time": time.time()}
    ws.latest_tickers.get.return_value = {"last": ws_last} if ws_last is not None else None

    with patch("api.services.trading_service.get_private_client", return_value=client), \
         patch("api.services.trading_service.get_public_client", return_value=client), \
         patch("api.services.trading_service.okx_ws_client", ws), \
         patch("api.services.trading_service.load_strategy",
               return_value={"formula": [0, 69], "formula_decoded": "RET -> NEG", "best_score": 2.5}), \
         patch("api.services.trading_service.eval_strategy_factor", return_value=torch.zeros(1, 800)), \
         patch("api.services.trading_service.compute_target_positions_stateless",
               return_value=torch.tensor([[signal]])):
        with patch.object(Config, "TRADING_MODE", "live"):
            res = service.execute_signal(
                strategy_path="dummy.json", inst_id=INST, capital=capital,
                leverage=5, bar="30m", max_position_pct=0.30,
            )
    return res, ws


# ── 1. 止损锚点 ────────────────────────────────────────────────────────────

def test_sl_anchors_to_live_ws_price(service):
    """收盘价 64,000、下单瞬间实时价 62,919：空头止损按 62,919 × 1.03 挂，而不是 64,000 × 1.03。"""
    client = MagicMock()
    res, _ = _run(service, client, signal=-0.25, ws_last=62919.0)

    assert client.place_order.call_args.kwargs["side"] == "sell"
    assert client.place_order.call_args.kwargs["sl_trigger_px"] == "64806.6"   # 62919 × 1.03
    assert res["order"]["sl_anchor_px"] == 62919.0
    assert res["order"]["sl_anchor_source"] == "ws"
    assert res["order"]["fill_avg_px"] == 64000.5


def test_stale_ws_ticker_is_skipped(service):
    """WS 连接已断 (或 30 秒无消息)：即使缓存里有价格也不用，改走 REST。"""
    client = MagicMock()
    client.get_ticker.return_value = {"last": "63100.0"}
    res, ws = _run(service, client, signal=-0.25, ws_last=61000.0, ws_fresh=False)

    ws.latest_tickers.get.assert_not_called()
    assert client.place_order.call_args.kwargs["sl_trigger_px"] == "64993.0"
    assert res["order"]["sl_anchor_source"] == "rest"


def test_sl_anchors_to_rest_price_when_ws_has_no_ticker(service):
    """WS 没有行情时退到 REST ticker。"""
    client = MagicMock()
    client.get_ticker.return_value = {"last": "63100.0"}
    res, _ = _run(service, client, signal=-0.25)

    assert client.place_order.call_args.kwargs["sl_trigger_px"] == "64993.0"   # 63100 × 1.03
    assert res["order"]["sl_anchor_source"] == "rest"


def test_sl_falls_back_to_close_when_live_price_missing_or_implausible(service):
    """WS 无价、REST 价格偏离收盘价 17% (多半是坏数据)：退回收盘价。"""
    client = MagicMock()
    client.get_ticker.return_value = {"last": "75000.0"}
    res, _ = _run(service, client, signal=-0.25, ws_last=0.0)

    assert client.place_order.call_args.kwargs["sl_trigger_px"] == "65920.0"   # 64000 × 1.03
    assert res["order"]["sl_anchor_source"] == "close"


def test_long_entry_sl_anchors_below_live_price(service):
    """多头开仓：止损挂在实时价下方 3%。"""
    client = MagicMock()
    res, _ = _run(service, client, signal=0.25, ws_last=61500.0)

    assert client.place_order.call_args.kwargs["side"] == "buy"
    assert client.place_order.call_args.kwargs["sl_trigger_px"] == "59655.0"   # 61500 × 0.97
    assert res["order"]["sl_anchor_source"] == "ws"


def test_sl_anchor_can_be_disabled(service):
    """关闭开关后恢复旧行为 (按收盘价)。"""
    client = MagicMock()
    with patch.object(Config, "SL_ANCHOR_TO_LIVE_PRICE", False):
        res, _ = _run(service, client, signal=-0.25, ws_last=62919.0)

    assert client.place_order.call_args.kwargs["sl_trigger_px"] == "65920.0"
    assert res["order"]["sl_anchor_source"] == "close"


# ── 2. 止损冷却 ────────────────────────────────────────────────────────────

def test_stop_cooldown_until_ends_at_bar_close():
    """bars=1 → 止损所在 K 线的收盘时刻；每多 1 根顺延一个周期。"""
    trig = dt.datetime(2026, 11, 23, 17, 8, 29, tzinfo=dt.timezone.utc).timestamp()
    fmt = lambda ts: dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%m-%d %H:%M")

    assert fmt(stop_cooldown_until(trig, 30, 1)) == "11-23 17:30"
    assert fmt(stop_cooldown_until(trig, 30, 2)) == "11-23 18:00"
    assert fmt(stop_cooldown_until(trig, 15, 1)) == "11-23 17:15"
    # 恰好落在边界上：算作下一根 K 线的开头
    on_boundary = dt.datetime(2026, 11, 23, 17, 30, tzinfo=dt.timezone.utc).timestamp()
    assert fmt(stop_cooldown_until(on_boundary, 30, 1)) == "11-23 18:00"
    assert stop_cooldown_until(trig, 30, 0) == 0.0


@pytest.mark.parametrize("target,current,blocked,expected", [
    (2.40, 0.0, +1, 0.0),     # 空仓想做多 (刚被止损的方向) → 不开
    (2.38, 1.0, +1, 1.0),     # 持多想加仓 → 维持现有仓位
    (1.00, 2.0, +1, 1.0),     # 持多想减仓 → 放行
    (0.00, 2.0, +1, 0.0),     # 持多想平仓 → 放行
    (-1.20, 0.0, +1, -1.2),   # 想做空 (反方向) → 放行
    (1.20, -0.5, +1, 0.0),    # 持空想翻多 → 只平空，不翻多
    (2.40, 0.0, -1, 2.4),     # 被止损的是空头 → 做多不受影响
    (-0.05, 0.0, -1, 0.0),    # 被止损的是空头 → 不再做空
])
def test_clamp_target_for_stop_cooldown(target, current, blocked, expected):
    assert clamp_target_for_stop_cooldown(target, current, blocked) == expected


def test_exchange_stop_blocks_same_direction_reentry(service):
    """空头刚被交易所端止损 (buy 止损单触发)，信号仍为空：不开仓。"""
    client = MagicMock()
    res, _ = _run(service, client, signal=-0.25, history=[_stopped_algo("buy", _ts_in_current_bar())])

    client.place_order.assert_not_called()
    assert res["target_held_sz"] == 0.0
    cooldown = res["stop_cooldown"]
    assert cooldown["blocked"] is True
    assert cooldown["direction"] == "short"
    assert cooldown["source"] == "exchange_sl"
    assert cooldown["original_target_sz"] == -5.85
    assert cooldown["adjusted_target_sz"] == 0.0


def test_exchange_stop_does_not_affect_opposite_direction(service):
    """空头刚被止损，但信号是多：照常开多。"""
    client = MagicMock()
    res, _ = _run(service, client, signal=0.25, history=[_stopped_algo("buy", _ts_in_current_bar())])

    assert client.place_order.call_count == 1
    assert client.place_order.call_args.kwargs["side"] == "buy"
    assert res["stop_cooldown"]["blocked"] is False


def test_stop_block_holds_position_without_adding(service):
    """多头被止损后仍有剩余仓位 (部分批次被扫)，信号要加仓：只维持，不加。"""
    client = MagicMock()
    res, _ = _run(service, client, signal=0.25, current_sz=5.0,
                  history=[_stopped_algo("sell", _ts_in_current_bar())])

    client.place_order.assert_not_called()
    assert res["order"]["skipped"] is True
    assert res["target_held_sz"] == 5.0
    assert res["stop_cooldown"]["blocked"] is True
    assert res["stop_cooldown"]["direction"] == "long"


def test_stop_block_allows_reduction_and_close(service):
    """冷却期内减仓照常执行 (减仓单不带止损单)。"""
    client = MagicMock()
    res, _ = _run(service, client, signal=0.12, current_sz=5.85,
                  history=[_stopped_algo("sell", _ts_in_current_bar())])

    assert client.place_order.call_count == 1
    assert client.place_order.call_args.kwargs["side"] == "sell"
    assert client.place_order.call_args.kwargs["sl_trigger_px"] is None
    assert res["stop_cooldown"]["blocked"] is False


def test_expired_stop_does_not_block(service):
    """两个周期前触发的止损，禁令早已结束。"""
    client = MagicMock()
    now = time.time()
    two_bars_ago = (math.floor(now / BAR_SEC) - 2) * BAR_SEC + 300
    res, _ = _run(service, client, signal=0.25, history=[_stopped_algo("sell", two_bars_ago)])

    assert client.place_order.call_count == 1
    assert res["stop_cooldown"]["blocked"] is False


def test_stop_algo_not_placed_by_this_bot_is_ignored(service):
    """不是本机器人挂的止损单 (ID 不符合 sl+13 位时间戳)，不触发冷却。"""
    client = MagicMock()
    res, _ = _run(service, client, signal=0.25,
                  history=[_stopped_algo("sell", _ts_in_current_bar(), algo_id="manual-stop-001")])

    assert client.place_order.call_count == 1
    assert res["stop_cooldown"]["blocked"] is False


def test_cooldown_disabled_when_bars_zero(service):
    """STOP_REENTRY_COOLDOWN_BARS=0：不查交易所、不拦截。"""
    client = MagicMock()
    with patch.object(Config, "STOP_REENTRY_COOLDOWN_BARS", 0):
        res, _ = _run(service, client, signal=0.25,
                      history=[_stopped_algo("sell", _ts_in_current_bar())])

    client.get_algo_orders_history.assert_not_called()
    assert client.place_order.call_count == 1
    assert res["stop_cooldown"] == {"cooldown_bars": 0, "blocked": False}


def test_history_query_failure_does_not_block_trading(service):
    """交易所历史查询失败：跳过交易所侧检测，不阻断交易。"""
    client = MagicMock()
    client.get_algo_orders_history.side_effect = RuntimeError("timeout")

    res, _ = _run(service, client, signal=0.25)

    assert client.place_order.call_count == 1
    assert res["stop_cooldown"]["blocked"] is False


def test_locally_marked_stop_exit_blocks_reentry(service):
    """本地登记的止损出场 (如追踪止损) 同样生效。"""
    client = MagicMock()
    service._mark_stop_exit(INST, +1, "30m", "trailing_sl")

    res, _ = _run(service, client, signal=0.25)

    client.place_order.assert_not_called()
    assert res["stop_cooldown"]["blocked"] is True
    assert res["stop_cooldown"]["source"] == "trailing_sl"

    # 自动交易状态里能看到这条禁令
    service._auto_trade_state["inst_id"] = INST
    block = service.get_auto_trade_status()["stop_reentry_block"]
    assert block["direction"] == "long"
    assert block["remaining_seconds"] >= 0


def test_breakeven_exit_registers_stop_block(service):
    """保本止损平仓后登记同向重进禁令。"""
    service._ladder_tp_ratchet[INST] = 0.65
    client = MagicMock()
    client.get_positions_detail.return_value = [
        {"inst_id": INST, "pos_side": "net", "pos": 1.0, "avg_px": 64000.0}
    ]
    client.get_ticker.return_value = {"last": "63000.0"}
    ws = MagicMock()
    ws.latest_tickers.get.return_value = None

    with patch("api.services.trading_service.get_private_client", return_value=client), \
         patch("api.services.trading_service.get_public_client", return_value=client), \
         patch("api.services.trading_service.okx_ws_client", ws), \
         patch.object(Config, "TRADING_MODE", "live"):
        res = service.check_realtime_ladder_tp(INST)

    assert res is not None
    assert res["event"] == "breakeven_stop_loss_trigger"
    assert service._stop_exit_block[INST]["source"] == "breakeven_sl"
    assert service._stop_exit_block[INST]["direction"] == 1


# 实盘复盘 (8010, 2026-11-15 ~ 11-23) 中真实发生的"止损后立刻重进"：
# (止损触发 UTC 时刻, 止损单平仓方向, 随后那次重进的 UTC 时刻, 1 根 K 线冷却下是否应被拦)
# 时间取自 OKX 策略委托历史的 triggerTime / 成交流水 / 审计日志。
_INCIDENT_REPLAY = [
    ("2026-11-19T20:31:12", "sell", "2026-11-19T20:33:00", True),    # 止损 2 分钟后加回
    ("2026-11-19T21:21:22", "sell", "2026-11-19T21:33:00", False),   # 止损所在 K 线已收盘，放行
    ("2026-11-21T03:32:09", "sell", "2026-11-21T03:34:00", True),    # 止损 2 分钟后重进，再被扫
    ("2026-11-21T03:32:09", "sell", "2026-11-21T04:04:00", False),   # 下一根 K 线的重进，后来 +62.6
    ("2026-11-23T17:08:29", "sell", "2026-11-23T17:19:00", True),    # 止损 11 分钟后买回
    ("2026-11-23T19:41:49", "sell", "2026-11-23T19:49:00", True),    # 止损 8 分钟后买回
    ("2026-11-15T02:12:43", "buy",  "2026-11-15T02:16:00", True),    # 保本止损后 4 分钟重开，随后 -3.93
    ("2026-11-20T21:42:43", "buy",  "2026-11-20T22:04:00", False),   # 追踪止损，所在 K 线已收盘
]


def _ts(iso):
    return dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc).timestamp()


@pytest.mark.parametrize("trigger,side,reentry,should_block", _INCIDENT_REPLAY,
                         ids=[f"{t[5:16]}->{r[11:16]}" for t, _, r, _ in _INCIDENT_REPLAY])
def test_incident_replay_one_bar_cooldown(trigger, side, reentry, should_block):
    """默认 1 根 K 线冷却，在真实事件上的效果：8 次重进拦住 5 次。

    没拦住的两次都发生在止损所在 K 线收盘之后 (信号已基于完整 K 线重算)。
    """
    until = stop_cooldown_until(_ts(trigger), 30, 1)
    assert (_ts(reentry) < until) is should_block


def test_incident_replay_two_bars_would_delay_the_winner():
    """把冷却调到 2 根 K 线的代价：11-21 04:04 那次重进 (后来 +62.6) 也会被推迟。"""
    until = stop_cooldown_until(_ts("2026-11-21T03:32:09"), 30, 2)
    assert _ts("2026-11-21T04:04:00") < until <= _ts("2026-11-21T04:34:00")


# ── 3. 仓位规模 ────────────────────────────────────────────────────────────

def test_effective_capital_rules():
    # 取不到权益 → 本金
    eff, detail = compute_effective_capital(1000.0, None)
    assert eff == 1000.0 and detail["sizing_basis"] == "capital"
    # 已实现权益低于本金 → 按已实现权益 (只缩)
    eff, detail = compute_effective_capital(1000.0, 800.0)
    assert eff == 800.0 and detail["sizing_basis"] == "min(capital, realized_equity)"
    # 已实现权益高于本金 → 仍按本金 (不放)
    eff, detail = compute_effective_capital(1000.0, 5000.0)
    assert eff == 1000.0 and detail["sizing_basis"] == "capital"
    # 已实现权益低到不像真的 (不足本金 20%) → 不采用
    eff, detail = compute_effective_capital(1000.0, 50.0)
    assert eff == 1000.0 and "未采用" in detail["sizing_basis"]
    # 权益字段是坏数据 (非数字) → 本金
    eff, _ = compute_effective_capital(1000.0, "n/a", "n/a")
    assert eff == 1000.0


def test_effective_capital_excludes_unrealized_pnl():
    """仓位基准用已实现权益 (总权益 - 浮盈亏)：持仓浮亏不影响目标仓位，避免顺着价格追涨杀跌。"""
    # 浮亏 200：总权益 1,000，已实现 1,200 → 仍按本金 1,000
    eff, detail = compute_effective_capital(1000.0, 1000.0, -200.0)
    assert eff == 1000.0 and detail["realized_eq"] == 1200.0
    # 浮盈 500：总权益 1,500，已实现 1,000 → 按本金 1,000，浮盈不用于加仓
    eff, detail = compute_effective_capital(1000.0, 1500.0, 500.0)
    assert eff == 1000.0
    # 浮盈 500 但总权益只有 1,200：已实现 700 → 按 700
    eff, detail = compute_effective_capital(1000.0, 1200.0, 500.0)
    assert eff == 700.0 and detail["realized_eq"] == 700.0


def test_effective_capital_can_be_disabled():
    with patch.object(Config, "SIZE_ON_LIVE_EQUITY", False):
        eff, detail = compute_effective_capital(1000.0, 800.0)
    assert eff == 1000.0 and detail["account_eq"] is None


def test_sizing_uses_equity_when_below_capital(service):
    """本金参数 1,000、已实现权益 800：目标仓位按 800 算 (0.46 张)，而不是 1,000 (0.58 张)。"""
    client = MagicMock()
    res, _ = _run(service, client, signal=-0.25, capital=1000.0, eq=800.0)

    assert res["target_held_sz"] == -0.46
    assert res["effective_capital"] == 800.0
    assert res["size_detail"]["effective_capital"] == 800.0
    assert res["size_detail"]["capital_param"] == 1000.0


def test_sizing_never_scales_above_capital(service):
    """权益高于本金参数时不放大仓位。"""
    client = MagicMock()
    res, _ = _run(service, client, signal=-0.25, capital=1000.0, eq=5000.0)

    assert res["target_held_sz"] == -0.58
    assert res["effective_capital"] == 1000.0


def test_sizing_ignores_implausibly_low_equity(service):
    client = MagicMock()
    res, _ = _run(service, client, signal=-0.25, capital=1000.0, eq=50.0)

    assert res["target_held_sz"] == -0.58
    assert "未采用" in res["size_detail"]["sizing_basis"]


def test_sizing_can_be_disabled(service):
    client = MagicMock()
    with patch.object(Config, "SIZE_ON_LIVE_EQUITY", False):
        res, _ = _run(service, client, signal=-0.25, capital=1000.0, eq=800.0)

    assert res["target_held_sz"] == -0.58


def test_realized_equity_shrink_trims_existing_position(service):
    """亏损平仓后基准缩小，已有的超额仓位会在下一根 K 线被减到新目标 (减仓单不带止损)。"""
    client = MagicMock()
    # 按 10,000 开的 5.85 张多单，已实现权益降到 8,000 → 目标 4.68 张
    res, _ = _run(service, client, signal=0.25, current_sz=5.85, capital=10000.0, eq=8000.0)

    assert res["target_held_sz"] == 4.68
    assert client.place_order.call_count == 1
    assert client.place_order.call_args.kwargs["side"] == "sell"
    assert client.place_order.call_args.kwargs["sl_trigger_px"] is None


def test_unrealized_pnl_does_not_change_target(service):
    """持仓浮亏不改变目标仓位：同样的已实现权益，浮亏从 0 到 -3,000，目标都是 5.85 张。"""
    client = MagicMock()
    res_flat, _ = _run(service, client, signal=0.25, current_sz=5.85, capital=10000.0, eq=10000.0, upl=0.0)
    res_drawn, _ = _run(service, client, signal=0.25, current_sz=5.85, capital=10000.0, eq=7000.0, upl=-3000.0)

    assert res_flat["target_held_sz"] == res_drawn["target_held_sz"] == 5.85
    assert res_drawn["size_detail"]["realized_eq"] == 10000.0
    assert client.place_order.call_count == 0
