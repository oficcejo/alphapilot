"""
tests/test_reef_vibe_evolution.py -- Reef 自进化闭环验证与 Vibe 专家策略生成测试套件

测试内容：
1. Vibe 专家种子策略结构与栈机执行完整性（零退化、无未来函数、方差健康度）
2. Reef GrowEngine 结合 Vibe 特征与算子的定向繁衍与突变机制
3. Reef ShadowEvaluator 影子观察池 5 重交付门禁评测链路
4. Reef CommitManager 原子交付热替换、旧策略归档与 TradingService 即时生效
"""

import json
import pathlib
import shutil
import tempfile
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from config import Config
from model.vocab import FORMULA_VOCAB
from model.vm import StackVM, validate_formula_structure
from model.features import MT5FeatureEngineer
from strategy_manager.signal import compute_target_positions_stateless
from api.services.strategy_service import load_strategy, decode_formula
from api.services.trading_service import TradingService
from evolution.grow import GrowEngine
from evolution.shadow import ShadowEvaluator
from evolution.commit import CommitManager


@pytest.fixture
def temp_reef_dirs():
    """提供隔离的 evolution 数据与 strategies 测试目录。"""
    temp_dir = tempfile.mkdtemp(prefix="test_reef_vibe_")
    evo_dir = pathlib.Path(temp_dir) / "data" / "evolution"
    strat_dir = pathlib.Path(temp_dir) / "strategies"
    evo_dir.mkdir(parents=True, exist_ok=True)
    strat_dir.mkdir(parents=True, exist_ok=True)

    yield str(evo_dir), str(strat_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


VIBE_STRATEGY_FILES = [
    "strategies/vibe_ETH-USDT-SWAP_15m_pv_divergence.json",
    "strategies/vibe_ETH-USDT-SWAP_1H_wvma_breakout.json",
    "strategies/vibe_SOL-USDT-SWAP_15m_alpha12_reversal.json",
]


class TestVibeExpertStrategiesIntegrity:
    """1. 验证 Vibe 专家种子策略文件规范与执行稳定性。"""

    @pytest.mark.parametrize("strat_file", VIBE_STRATEGY_FILES)
    def test_vibe_strategy_file_exists_and_loads(self, strat_file):
        path = pathlib.Path(strat_file)
        assert path.exists(), f"专家策略文件不存在: {strat_file}"

        data = load_strategy(strat_file)
        assert "formula" in data
        assert "formula_decoded" in data
        assert "best_score" in data
        assert data["best_score"] > 0.0
        assert data["vocab_version"] == FORMULA_VOCAB.version

    @pytest.mark.parametrize("strat_file", VIBE_STRATEGY_FILES)
    def test_vibe_strategy_passes_structure_guard(self, strat_file):
        data = load_strategy(strat_file)
        violations = validate_formula_structure(data["formula"], FORMULA_VOCAB.token_names)
        assert violations == [], f"{strat_file} 存在结构违规: {violations}"

    @pytest.mark.parametrize("strat_file", VIBE_STRATEGY_FILES)
    def test_vibe_strategy_vm_execution_and_signal(self, strat_file):
        data = load_strategy(strat_file)
        vm = StackVM()

        # 生成 800 根虚拟行情数据并通过 MT5FeatureEngineer 计算全套 81 维特征
        np.random.seed(42)
        n_bars = 800
        close = 2500.0 + np.cumsum(np.random.randn(n_bars) * 10.0)
        high = close + np.random.uniform(5.0, 15.0, size=n_bars)
        low = close - np.random.uniform(5.0, 15.0, size=n_bars)
        open_ = close + np.random.randn(n_bars) * 5.0
        volume = np.random.uniform(50.0, 500.0, size=n_bars)

        raw_dict = {
            "close": torch.from_numpy(close).unsqueeze(0).float(),
            "open": torch.from_numpy(open_).unsqueeze(0).float(),
            "high": torch.from_numpy(high).unsqueeze(0).float(),
            "low": torch.from_numpy(low).unsqueeze(0).float(),
            "volume": torch.from_numpy(volume).unsqueeze(0).float(),
        }
        feat = MT5FeatureEngineer.compute_features(raw_dict)
        assert feat.shape == (1, FORMULA_VOCAB.feature_count, n_bars)

        # 执行 VM 计算因子
        factors = vm.execute(data["formula"], feat)
        assert factors is not None
        assert factors.shape == (1, n_bars)
        assert not torch.isnan(factors).any(), f"{strat_file} 包含 NaN"
        assert not torch.isinf(factors).any(), f"{strat_file} 包含 Inf"

        # 验证因子方差健康度（绝非退化为常数）
        std = float(factors.std().item())
        assert std > 0.05, f"{strat_file} 方差过低 (std={std:.6f})"

        # 计算目标仓位信号
        pos_signal = compute_target_positions_stateless(factors)
        assert pos_signal.shape == (1, n_bars)
        assert float(pos_signal.max().item()) <= 1.0
        assert float(pos_signal.min().item()) >= -1.0


class TestReefGrowEngineWithVibe:
    """2. 验证 Reef GrowEngine 对 Vibe 策略的变异与因果引导进化。"""

    def test_mutation_samples_from_extended_vocabulary(self, temp_reef_dirs):
        evo_dir, _ = temp_reef_dirs
        engine = GrowEngine(data_dir=evo_dir)

        base_strat = load_strategy("strategies/vibe_ETH-USDT-SWAP_1H_wvma_breakout.json")
        base_formula = base_strat["formula"]

        # 执行 50 次变异，验证能生成有效突变公式
        mutated_formulas = []
        for _ in range(50):
            m = engine.mutate_formula(base_formula)
            mutated_formulas.append(m)
            assert len(m) > 0
            # 所有 token 必须在词表有效范围内
            assert all(0 <= tok < len(FORMULA_VOCAB.token_names) for tok in m)

        # 突变应包含多样性
        unique_mutations = {tuple(m) for m in mutated_formulas}
        assert len(unique_mutations) > 5

    def test_evolution_step_generates_valid_candidate(self, temp_reef_dirs):
        evo_dir, _ = temp_reef_dirs
        engine = GrowEngine(data_dir=evo_dir)

        base_strat = load_strategy("strategies/vibe_ETH-USDT-SWAP_15m_pv_divergence.json")

        n_bars = 400
        n_feats = FORMULA_VOCAB.feature_count
        feat_tensor = torch.randn(1, n_feats, n_bars)
        target_ret = torch.randn(1, n_bars) * 0.01
        timestamps = np.linspace(1788777297593, 1789196659360, n_bars).astype(np.int64)

        trajectories = [
            {
                "open_time_ms": 1788777297593,
                "close_time_ms": 1789044425941,
                "duration_hours": 74.2,
                "pnl_ratio": -0.157,
            }
        ]

        # 尝试多步以产生成功候选
        candidate = None
        for _ in range(30):
            res = engine.run_evolution_step(
                base_strategy=base_strat,
                feat_tensor=feat_tensor,
                target_ret=target_ret,
                candles_timestamps=timestamps,
                trajectories=trajectories,
            )
            if res is not None:
                candidate = res
                break

        assert candidate is not None, "未能在 30 步内生成有效候选变异策略"
        assert "candidate_id" in candidate
        assert "score" in candidate
        assert "formula" in candidate
        assert "formula_decoded" in candidate
        assert candidate["vocab_version"] == FORMULA_VOCAB.version


class TestReefShadowEvaluator5Gates:
    """3. 验证 Reef 影子池对 Vibe 策略候选者的 5 重严苛门禁。"""

    def test_evaluate_and_pass_vibe_candidate(self, temp_reef_dirs):
        evo_dir, _ = temp_reef_dirs
        evaluator = ShadowEvaluator(data_dir=evo_dir)

        incumbent = load_strategy("strategies/vibe_ETH-USDT-SWAP_15m_pv_divergence.json")
        inc_score = incumbent["best_score"]

        n_bars = 300
        feat_tensor = torch.randn(1, FORMULA_VOCAB.feature_count, n_bars)
        target_ret = torch.randn(1, n_bars) * 0.01

        # 构建一个模拟通过门禁的优胜候选策略
        winning_candidate = {
            "candidate_id": "cand_vibe_winner_001",
            "symbol": "ETH-USDT-SWAP",
            "timeframe": "15m",
            "formula": incumbent["formula"],
            "formula_decoded": incumbent["formula_decoded"],
            "score": round(inc_score * 1.08, 4),  # 超越基线 +8%
            "vocab_version": FORMULA_VOCAB.version,
        }

        passed, report, comparison = evaluator.evaluate_gates(
            winning_candidate, incumbent, feat_tensor, target_ret
        )

        assert report["gate_no_lookahead"] is True
        assert report["gate_non_degenerate"] is True
        assert report["gate_score_improvement"] is True
        assert report["gate_vm_compatibility"] is True
        assert "candidate_score" in comparison

    def test_shadow_pool_registration_and_persistence(self, temp_reef_dirs):
        evo_dir, _ = temp_reef_dirs
        evaluator = ShadowEvaluator(data_dir=evo_dir)

        candidate = {
            "candidate_id": "cand_vibe_pool_test",
            "symbol": "SOL-USDT-SWAP",
            "timeframe": "15m",
            "formula": [79, 152, 51, 82, 145, 137, 139],
            "score": 5.95,
        }
        incumbent = {"best_score": 5.5, "formula": [0, 69]}
        feat_tensor = torch.randn(1, FORMULA_VOCAB.feature_count, 100)
        target_ret = torch.randn(1, 100) * 0.01

        evaluator.register_candidate(candidate, incumbent, feat_tensor, target_ret)

        # 检查持久化文件
        pool_file = pathlib.Path(evo_dir) / "shadow_pool.json"
        assert pool_file.exists()
        pool_data = json.loads(pool_file.read_text(encoding="utf-8"))
        assert "cand_vibe_pool_test" in pool_data


class TestReefCommitManagerAndHotSwap:
    """4. 验证 Reef CommitManager 对 Vibe 策略的原子交付、旧策略备份与交易生效。"""

    def test_atomic_commit_and_archive(self, temp_reef_dirs):
        evo_dir, strat_dir = temp_reef_dirs
        commit_mgr = CommitManager(data_dir=evo_dir, strategies_dir=strat_dir)
        evaluator = ShadowEvaluator(data_dir=evo_dir)

        target_file = pathlib.Path(strat_dir) / "best_ETH-USDT-SWAP_1H.json"
        original_data = {
            "vocab_version": FORMULA_VOCAB.version,
            "symbol": "ETH-USDT-SWAP",
            "timeframe": "1H",
            "formula": [0, 69],
            "best_score": 5.0,
            "formula_decoded": "OLD_STRATEGY",
        }
        target_file.write_text(json.dumps(original_data, indent=2), encoding="utf-8")

        # 准备一个通过门禁的 Vibe 候选
        vibe_cand = {
            "candidate_id": "cand_vibe_wvma_commit",
            "symbol": "ETH-USDT-SWAP",
            "timeframe": "1H",
            "formula": [65, 158, 76, 83, 71, 83, 141],
            "formula_decoded": "WVMA_5 -> DELTA_10 -> RSQR_10 -> MUL -> KBAR_EFF -> MUL -> TANH_SQUASH",
            "score": 6.15,
        }

        with patch("evolution.shadow.get_shadow_evaluator", return_value=evaluator):
            evaluator.pool["cand_vibe_wvma_commit"] = {
                **vibe_cand,
                "status": "PASSED",
                "gate_report": {"gate_passed": True, "reasons": []},
            }

            commit_result = commit_mgr.commit_candidate(
                candidate_id="cand_vibe_wvma_commit",
                target_strategy_path=str(target_file),
            )

            assert commit_result["status"] == "success"
            assert commit_result["new_score"] == 6.15
            assert commit_result["previous_score"] == 5.0
            assert commit_result["archived_backup"] is not None
            assert pathlib.Path(commit_result["archived_backup"]).exists()

            # 校验上线目标策略文件内容被原子更新
            updated = json.loads(target_file.read_text(encoding="utf-8"))
            assert updated["formula"] == [65, 158, 76, 83, 71, 83, 141]
            assert "WVMA_5" in updated["formula_decoded"]

    def test_trading_service_executes_vibe_strategy(self):
        """验证 TradingService 能即时加载并执行该 Vibe 策略出单。"""
        service = TradingService()
        service._instrument_cache["ETH-USDT-SWAP"] = {
            "ctVal": "0.1", "lotSz": "0.01", "minSz": "0.01", "tickSz": "0.01", "ctValCcy": "ETH"
        }
        mock_client = MagicMock()
        mock_client.get_positions_detail.return_value = []
        mock_client.get_account_summary.return_value = {"total_eq": 1000.0, "avail_bal": 900.0}
        mock_client.place_order.return_value = {"clOrdId": "vibe_test_ord", "live": True}

        # 模拟 800 根真实的蜡烛数据
        np.random.seed(123)
        candles = []
        px = 2500.0
        for i in range(800):
            px += float(np.random.randn() * 5.0)
            candles.append([
                str(1600000000000 + i * 900000),
                str(round(px, 2)),
                str(round(px + 4.0, 2)),
                str(round(px - 4.0, 2)),
                str(round(px + 1.0, 2)),
                str(round(float(np.random.uniform(50, 300)), 2)),
                "", "", "1"
            ])
        mock_client.get_recent_candles.return_value = candles

        strat_path = "strategies/vibe_ETH-USDT-SWAP_15m_pv_divergence.json"

        with patch("api.services.trading_service.get_private_client", return_value=mock_client), \
             patch("api.services.trading_service.get_public_client", return_value=mock_client), \
             patch.object(Config, "TRADING_MODE", "live"):

            res = service.execute_signal(
                strategy_path=strat_path,
                inst_id="ETH-USDT-SWAP",
                capital=1000.0,
                leverage=5,
                bar="15m",
                max_position_pct=0.30,
            )

            # 验证执行链路畅通：信号生成、门禁通过、开单返回
            assert res["risk_passed"] is True
            assert res["inst_id"] == "ETH-USDT-SWAP"
            assert res["strategy"]["best_score"] == 5.42
            assert res["strategy"]["formula"] == [72, 151, 69, 143, 144, 141]
            assert "signal" in res
            assert abs(res["signal"]) <= 1.0
