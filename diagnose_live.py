#!/usr/bin/env python3
"""诊断脚本：查询OKX历史仓位 + 分析策略信号问题"""
import hmac, base64, hashlib, time, json, requests, os, sys

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('OKX_API_KEY', '')
API_SECRET = os.getenv('OKX_API_SECRET', '')
PASSPHRASE = os.getenv('OKX_API_PASSPHRASE', '')
BASE_URL = os.getenv('OKX_API_BASE', 'https://www.okx.com')
SIMULATED = os.getenv('OKX_SIMULATED', '0')


def timestamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())


def sign(ts, method, path, body=''):
    msg = f'{ts}{method.upper()}{path}{body}'
    mac = hmac.new(API_SECRET.encode(), msg.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()


def auth_headers(method, path, body=''):
    ts = timestamp()
    return {
        'OK-ACCESS-KEY': API_KEY,
        'OK-ACCESS-SIGN': sign(ts, method, path, body),
        'OK-ACCESS-TIMESTAMP': ts,
        'OK-ACCESS-PASSPHRASE': PASSPHRASE,
        'x-simulated-trading': SIMULATED,
        'Content-Type': 'application/json',
    }


def query_account_balance():
    """查询账户余额"""
    path = '/api/v5/account/balance'
    headers = auth_headers('GET', path)
    resp = requests.get(BASE_URL + path, headers=headers, timeout=15)
    data = resp.json()
    print('=== 账户余额 ===')
    if data.get('code') == '0':
        acct = data.get('data', [{}])[0]
        print(f'  总权益: {acct.get("totalEq")} USDT')
        for d in acct.get('details', []):
            if d.get('ccy') == 'USDT':
                print(f'  USDT: 权益={d.get("eq")}  可用={d.get("availBal")}  '
                      f'未实现PnL={d.get("upl")}  已实现PnL={d.get("realizedPnl")}')
        return acct
    else:
        print(f'  Error: {data.get("msg", data)}')
        return None


def query_positions_history(limit=50, inst_id=None):
    """查询历史仓位"""
    params = f'instType=SWAP&limit={limit}'
    if inst_id:
        params += f'&instId={inst_id}'
    path = f'/api/v5/account/positions-history?{params}'
    headers = auth_headers('GET', path)
    resp = requests.get(BASE_URL + path, headers=headers, timeout=15)
    data = resp.json()
    print(f'\n=== 历史仓位 (limit={limit}) ===')
    if data.get('code') == '0':
        positions = data.get('data', [])
        print(f'  共 {len(positions)} 条记录')
        type_map = {'1': '开仓', '2': '平仓', '3': '部分平仓', '4': '强平'}
        total_pnl = 0.0
        for p in positions:
            inst = p.get('instId', '')
            pos_side = p.get('posSide', '')
            type_ = type_map.get(p.get('type', ''), p.get('type', ''))
            realized_pnl = float(p.get('realizedPnl', '0') or '0')
            pnl_ratio = p.get('pnlRoi', '0')
            margin = p.get('margin', '0')
            lever = p.get('lever', '')
            open_px = p.get('openAvgPx', '')
            close_px = p.get('closeAvgPx', '')
            c_time = p.get('cTime', '')
            u_time = p.get('uTime', '')
            pos = p.get('pos', '')
            close_pos = p.get('closePos', '')
            fee = p.get('fee', '0')
            funding_fee = p.get('fundingFee', '0')
            total_pnl += realized_pnl
            print(f'  {inst} | {pos_side} {type_} | pos={pos} closePos={close_pos} | '
                  f'PnL={realized_pnl:.4f} ROI={pnl_ratio} | lever={lever} margin={margin} | '
                  f'open={open_px} close={close_px} | fee={fee} funding={funding_fee} | '
                  f'cTime={c_time} uTime={u_time}')
        print(f'\n  累计已实现PnL: {total_pnl:.4f} USDT')
        return positions
    else:
        print(f'  Error: {data.get("msg", data)}')
        return []


def query_current_positions():
    """查询当前持仓"""
    path = '/api/v5/account/positions'
    headers = auth_headers('GET', path)
    resp = requests.get(BASE_URL + path, headers=headers, timeout=15)
    data = resp.json()
    print('\n=== 当前持仓 ===')
    if data.get('code') == '0':
        positions = data.get('data', [])
        if not positions:
            print('  无持仓')
        for p in positions:
            print(f'  {p.get("instId")} side={p.get("posSide")} pos={p.get("pos")} '
                  f'upl={p.get("upl")} uplRatio={p.get("uplRatio")} '
                  f'lever={p.get("lever")} margin={p.get("margin")} '
                  f'liqPx={p.get("liqPx")} mgnMode={p.get("mgnMode")}')
        return positions
    else:
        print(f'  Error: {data.get("msg", data)}')
        return []


def query_fills(limit=50):
    """查询最近成交记录"""
    path = f'/api/v5/trade/fills?instType=SWAP&limit={limit}'
    headers = auth_headers('GET', path)
    resp = requests.get(BASE_URL + path, headers=headers, timeout=15)
    data = resp.json()
    print(f'\n=== 最近成交记录 (limit={limit}) ===')
    if data.get('code') == '0':
        fills = data.get('data', [])
        print(f'  共 {len(fills)} 条记录')
        total_fee = 0.0
        for f in fills[:30]:
            fee = float(f.get('fee', '0') or '0')
            total_fee += fee
            print(f'  {f.get("instId")} side={f.get("side")} posSide={f.get("posSide")} '
                  f'sz={f.get("sz")} px={f.get("fillPx")} fee={f.get("fee")} '
                  f'feeCcy={f.get("feeCcy")} ts={f.get("ts")} clOrdId={f.get("clOrdId","")}')
        print(f'\n  累计手续费: {total_fee:.4f} USDT')
        return fills
    else:
        print(f'  Error: {data.get("msg", data)}')
        return []


def query_account_config():
    """查询账户配置"""
    path = '/api/v5/account/config'
    headers = auth_headers('GET', path)
    resp = requests.get(BASE_URL + path, headers=headers, timeout=15)
    data = resp.json()
    print('\n=== 账户配置 ===')
    if data.get('code') == '0':
        cfg = data.get('data', [{}])[0]
        print(f'  持仓模式: {cfg.get("posMode")}')
        print(f'  账户级别: {cfg.get("acctLv")}')
        print(f'  保证金模式: {cfg.get("mgnMode")}')
        return cfg
    else:
        print(f'  Error: {data.get("msg", data)}')
        return None


def analyze_strategy_signal():
    """分析策略信号生成逻辑"""
    print('\n' + '='*60)
    print('=== 策略公式分析 ===')
    print('='*60)

    # 加载策略
    with open('strategies/best_ETH-USDT-SWAP.json', 'r') as f:
        strategy = json.load(f)

    formula = strategy.get('formula', [])
    print(f'\n策略公式 tokens: {formula}')
    print(f'best_score: {strategy.get("best_score")}')

    # 解码公式
    from model.vocab import FORMULA_VOCAB
    names = FORMULA_VOCAB.token_names
    feat_offset = FORMULA_VOCAB.operator_offset
    print(f'\n词表: {len(names)} tokens (features={FORMULA_VOCAB.feature_count}, operators={len(FORMULA_VOCAB.operator_names)})')
    print(f'feat_offset: {feat_offset}')

    decoded = []
    for t in formula:
        t = int(t)
        if t < feat_offset:
            decoded.append(f'FEAT[{t}]={names[t]}')
        else:
            op_idx = t - feat_offset
            from model.ops import OPS_CONFIG
            op_name = OPS_CONFIG[op_idx][0]
            op_arity = OPS_CONFIG[op_idx][2]
            decoded.append(f'OP[{op_idx}]={op_name}(arity={op_arity})')
    print(f'\n解码:')
    for d in decoded:
        print(f'  {d}')

    # 追踪栈执行
    print('\n=== 栈执行追踪 ===')
    from model.vm import StackVM
    vm = StackVM()
    stack_depth = 0
    trace = []
    for i, token in enumerate(formula):
        token = int(token)
        if token < vm.feat_offset:
            stack_depth += 1
            trace.append(f'Step {i}: PUSH {names[token]} -> depth={stack_depth}')
        elif token in vm.op_map:
            arity = vm.arity_map[token]
            op_name = names[token]
            stack_depth -= arity - 1
            trace.append(f'Step {i}: CALL {op_name}(arity={arity}) -> depth={stack_depth}')
    for t in trace:
        print(f'  {t}')
    print(f'  最终栈深度: {stack_depth} (需要=1)')

    # 实际执行并分析信号
    print('\n=== 实盘信号模拟 ===')
    import torch
    import numpy as np
    from data_pipeline.okx_client import get_public_client
    from model.features import MT5FeatureEngineer
    from strategy_manager.signal import compute_target_positions_stateless, LOWER_BAND, UPPER_BAND

    client = get_public_client()

    # 测试 5m 和 1H
    for bar in ['5m', '1H']:
        print(f'\n--- {bar} 周期 ---')
        candles = client.get_candles('ETH-USDT-SWAP', bar, limit=300)
        if not candles:
            print(f'  未获取到 {bar} 行情')
            continue

        close_arr = np.array([float(c[4]) for c in candles], dtype=np.float64)
        open_arr = np.array([float(c[1]) for c in candles], dtype=np.float64)
        high_arr = np.array([float(c[2]) for c in candles], dtype=np.float64)
        low_arr = np.array([float(c[3]) for c in candles], dtype=np.float64)
        vol_arr = np.array([float(c[5]) for c in candles], dtype=np.float64)
        time_arr = np.array([int(c[0]) // 1000 if int(c[0]) > 1e12 else int(c[0]) for c in candles], dtype=np.float64)

        raw_dict = {
            'close': torch.from_numpy(close_arr).unsqueeze(0).float(),
            'open': torch.from_numpy(open_arr).unsqueeze(0).float(),
            'high': torch.from_numpy(high_arr).unsqueeze(0).float(),
            'low': torch.from_numpy(low_arr).unsqueeze(0).float(),
            'volume': torch.from_numpy(vol_arr).unsqueeze(0).float(),
            'time': time_arr,
        }

        feat = MT5FeatureEngineer.compute_features(raw_dict)
        with torch.no_grad():
            factor = vm.execute(formula, feat)

        if factor is None:
            print(f'  因子执行失败 (None)')
            continue

        position_signal = compute_target_positions_stateless(factor)
        signal = float(position_signal[0, -1].item())
        factor_val = float(factor[0, -1].item())
        tanh_val = float(torch.tanh(torch.tensor(factor_val)).item())

        print(f'  K线数量: {len(close_arr)}')
        print(f'  最新价格: {close_arr[-1]}')
        print(f'  因子值 (最后10): {[round(float(factor[0, -i].item()), 4) for i in range(10, 0, -1)]}')
        print(f'  tanh(因子): {tanh_val:.6f}')
        print(f'  |tanh|: {abs(tanh_val):.6f}')
        print(f'  Neutral Band: [{LOWER_BAND}, {UPPER_BAND}]')
        print(f'  信号值: {signal:.6f}')
        print(f'  信号阈值: 0.05 (低于此值不交易)')
        print(f'  是否触发交易: {"是" if abs(signal) >= 0.05 else "否"}')

        # 统计最近100根K线的信号分布
        recent_signals = [float(position_signal[0, i].item()) for i in range(max(0, len(close_arr)-100), len(close_arr))]
        non_zero = [s for s in recent_signals if abs(s) >= 0.05]
        print(f'  最近100根K线信号统计:')
        print(f'    有效信号(>=0.05): {len(non_zero)}/100')
        print(f'    信号均值: {np.mean(recent_signals):.4f}')
        print(f'    信号最大值: {max(recent_signals):.4f}')
        print(f'    信号最小值: {min(recent_signals):.4f}')
        print(f'    信号标准差: {np.std(recent_signals):.4f}')

        # 计算理论年化收益
        from data_pipeline.timeframe_utils import infer_periods_per_year
        periods_per_year = infer_periods_per_year(time_arr)
        print(f'  每年周期数: {periods_per_year}')

        # 回测成本对比
        bt_cost = 0.0001  # 回测使用
        live_cost = 0.0005 + 0.0003  # 手续费 + 滑点
        print(f'  回测成本率: {bt_cost} (0.01%)')
        print(f'  实盘成本率: {live_cost} (0.08%)')
        print(f'  成本差异: {live_cost/bt_cost:.1f}倍')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print('='*60)
    print('  OKX 实盘诊断 — ETH-USDT-SWAP 5m 策略')
    print('='*60)
    print(f'  交易模式: {"实盘" if SIMULATED == "0" else "模拟盘"}')
    print(f'  API Base: {BASE_URL}')
    print()

    # 1. 账户配置
    query_account_config()

    # 2. 账户余额
    query_account_balance()

    # 3. 当前持仓
    query_current_positions()

    # 4. 历史仓位
    query_positions_history(limit=50, inst_id='ETH-USDT-SWAP')

    # 5. 成交记录
    query_fills(limit=50)

    # 6. 策略分析
    analyze_strategy_signal()

    print('\n' + '='*60)
    print('  诊断完成')
    print('='*60)
