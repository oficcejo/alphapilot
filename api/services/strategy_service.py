"""
api/services/strategy_service.py — 策略管理与多因子组合服务

加载、保存、解码策略 JSON 文件，以及多因子组合策略构建与评估。
"""
import json
import pathlib
import math
from typing import Optional, List, Dict, Any
import torch

from config import Config
from model.vocab import FORMULA_VOCAB, VOCAB_VERSION
from model.vm import StackVM


def strategies_dir() -> pathlib.Path:
    d = pathlib.Path(Config.STRATEGIES_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_strategies() -> list[dict]:
    """列出所有已保存的策略（包含单因子策略与组合策略）。"""
    files = sorted(strategies_dir().glob("*.json"))
    result = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            is_portfolio = data.get("is_portfolio", False)

            if is_portfolio:
                sub_strats = data.get("strategies", [])
                sub_names = [s.get("file_name", "") for s in sub_strats]
                result.append({
                    "file_name": f.name,
                    "file_path": f.as_posix(),
                    "is_portfolio": True,
                    "portfolio_name": data.get("portfolio_name", f.stem),
                    "weight_method": data.get("weight_method", "equal"),
                    "symbol": data.get("symbol") or "组合",
                    "formula": None,
                    "formula_decoded": f"[组合 {len(sub_strats)} 策略]: " + ", ".join(sub_names[:3]) + ("..." if len(sub_names) > 3 else ""),
                    "best_score": data.get("best_score", 0),
                    "vocab_version": data.get("vocab_version", ""),
                    "data_file": data.get("data_file"),
                    "timeframe": data.get("timeframe"),
                    "sub_strategies": sub_strats,
                })
            else:
                result.append({
                    "file_name": f.name,
                    "file_path": f.as_posix(),
                    "is_portfolio": False,
                    "symbol": data.get("symbol"),
                    "formula": data.get("formula"),
                    "formula_decoded": data.get("formula_decoded", decode_formula(data.get("formula"))),
                    "best_score": data.get("best_score", 0),
                    "vocab_version": data.get("vocab_version", ""),
                    "data_file": data.get("data_file"),
                    "timeframe": data.get("timeframe"),
                    "mode": data.get("mode"),
                    "train_steps": data.get("train_steps"),
                })
        except Exception as e:
            result.append({"file_name": f.name, "error": str(e)})
    return result


def load_strategy(path: str) -> dict:
    """加载策略 JSON（支持单策略与组合策略）。"""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"策略文件不存在: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not data.get("is_portfolio") and "formula_decoded" not in data:
        data["formula_decoded"] = decode_formula(data.get("formula"))
    return data


def decode_formula(tokens: Optional[list[int]]) -> str:
    """将 token 列表解码为可读公式字符串。"""
    if not tokens:
        return "无"
    names = FORMULA_VOCAB.token_names
    parts = []
    for t in tokens:
        t = int(t)
        if 0 <= t < len(names):
            parts.append(names[t])
        else:
            parts.append(f"?{t}")
    return " → ".join(parts)


def save_strategy(formula: list[int], score: float, symbol: Optional[str] = None,
                  data_file: Optional[str] = None, timeframe: Optional[str] = None) -> str:
    """保存单因子策略到 JSON。"""
    parts = []
    if symbol:
        parts.append(symbol)
    if timeframe:
        tf_safe = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(timeframe))
        parts.append(tf_safe)
    tag = "_".join(parts) if parts else ""
    path = strategies_dir() / (f"best_{tag}.json" if tag else "best.json")
    data = {
        "vocab_version": VOCAB_VERSION,
        "is_portfolio": False,
        "symbol": symbol,
        "timeframe": timeframe,
        "data_file": data_file,
        "formula": formula,
        "formula_decoded": decode_formula(formula),
        "best_score": score,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def create_portfolio_strategy(
    portfolio_name: str,
    strategy_paths: List[str],
    weight_method: str = "score_weighted",
    custom_weights: Optional[List[float]] = None,
) -> dict:
    """构建多因子组合策略并保存 JSON。

    Args:
        portfolio_name: 组合策略名称
        strategy_paths: 参与融合的子策略路径列表
        weight_method: "equal"（等权）/ "score_weighted"（按得分加权）/ "manual"（自定义）
        custom_weights: 自定义权重列表
    """
    if not strategy_paths:
        raise ValueError("请至少选择一个子策略进行融合")

    sub_strategies = []
    scores = []
    symbol = None
    timeframe = None

    for path in strategy_paths:
        strat = load_strategy(path)
        if strat.get("is_portfolio"):
            raise ValueError(f"无法将已有组合策略 {path} 嵌套嵌套为子策略")
        score = float(strat.get("best_score", 1.0))
        scores.append(max(0.01, score))
        if not symbol and strat.get("symbol"):
            symbol = strat.get("symbol")
        if not timeframe and strat.get("timeframe"):
            timeframe = strat.get("timeframe")

        sub_strategies.append({
            "file_name": pathlib.Path(path).name,
            "file_path": path,
            "symbol": strat.get("symbol"),
            "formula": strat.get("formula"),
            "formula_decoded": strat.get("formula_decoded"),
            "best_score": score,
        })

    n = len(sub_strategies)
    if weight_method == "equal":
        weights = [1.0 / n] * n
    elif weight_method == "manual" and custom_weights and len(custom_weights) == n:
        total_w = sum(custom_weights) or 1.0
        weights = [w / total_w for w in custom_weights]
    else:
        # score_weighted: 按 best_score 比例加权
        total_s = sum(scores) or 1.0
        weights = [s / total_s for s in scores]

    for i in range(n):
        sub_strategies[i]["weight"] = round(weights[i], 4)

    # 安全文件名
    clean_name = ''.join(c if (c.isalnum() or c in ('-', '_')) else '_' for c in portfolio_name)
    if not clean_name:
        clean_name = "multi_factor"
    file_path = strategies_dir() / f"portfolio_{clean_name}.json"

    avg_score = sum(scores) / n
    portfolio_data = {
        "is_portfolio": True,
        "portfolio_name": portfolio_name,
        "weight_method": weight_method,
        "symbol": symbol,
        "timeframe": timeframe,
        "vocab_version": VOCAB_VERSION,
        "best_score": round(avg_score, 4),
        "strategies": sub_strategies,
    }

    file_path.write_text(json.dumps(portfolio_data, indent=2, ensure_ascii=False), encoding="utf-8")
    portfolio_data["file_name"] = file_path.name
    portfolio_data["file_path"] = file_path.as_posix()
    return portfolio_data


def eval_strategy_factor(strategy: dict, vm: StackVM, feat_dict: torch.Tensor) -> torch.Tensor:
    """统一求值接口：计算单策略或组合策略的加权合成 Alpha 因子。

    Args:
        strategy: 策略字典（load_strategy 返回）
        vm: StackVM 实例
        feat_dict: 提取的特征张量 [N, num_features, T]

    Returns:
        因子张量 [N, T]
    """
    if strategy.get("is_portfolio"):
        sub_strats = strategy.get("strategies", [])
        if not sub_strats:
            raise ValueError("组合策略中没有子策略")

        composite_factor = None
        for sub in sub_strats:
            formula = sub.get("formula")
            weight = float(sub.get("weight", 1.0))
            if not formula:
                continue

            f_i = vm.execute(formula, feat_dict)
            if f_i is None:
                continue

            # 时序 Z-Score 标准化（防止某个因子的绝对数值量级过大主导全局）
            mean = f_i.mean(dim=-1, keepdim=True)
            std = f_i.std(dim=-1, keepdim=True)
            f_norm = (f_i - mean) / (std + 1e-6)

            if composite_factor is None:
                composite_factor = weight * f_norm
            else:
                composite_factor = composite_factor + weight * f_norm

        if composite_factor is None:
            raise ValueError("组合策略中的所有子策略算子求值均失败")
        return composite_factor
    else:
        formula = strategy.get("formula")
        if not formula:
            raise ValueError("单策略缺少公式 formula")
        f = vm.execute(formula, feat_dict)
        if f is None:
            raise ValueError("公式执行失败")
        return f


def delete_strategy(path: str) -> dict:
    """删除策略 JSON 文件。"""
    p = pathlib.Path(path).resolve()
    base = strategies_dir().resolve()

    try:
        p.relative_to(base)
    except ValueError:
        raise ValueError(f"非法路径：策略文件必须在 {base} 目录内")

    if not p.exists():
        raise FileNotFoundError(f"策略文件不存在: {path}")

    if not p.is_file():
        raise ValueError(f"目标不是文件: {path}")

    file_name = p.name
    p.unlink()

    return {"ok": True, "deleted": file_name}


def import_strategy(content_bytes: bytes, filename: str) -> dict:
    """导入外部策略 JSON 文件。"""
    if not content_bytes:
        raise ValueError("上传的文件内容为空")

    try:
        data = json.loads(content_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"无效的 JSON 格式: {e}")

    if not isinstance(data, dict):
        raise ValueError("策略 JSON 必须为对象格式")

    is_portfolio = data.get("is_portfolio", False)
    if not is_portfolio:
        formula = data.get("formula")
        if not formula or not isinstance(formula, list):
            raise ValueError("策略文件缺少有效的 'formula' 算子序列")

    p_name = pathlib.Path(filename).name
    safe_name = "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in p_name)
    if not safe_name.endswith(".json"):
        safe_name += ".json"

    if not is_portfolio and "formula_decoded" not in data:
        data["formula_decoded"] = decode_formula(data.get("formula"))
    if "vocab_version" not in data:
        data["vocab_version"] = VOCAB_VERSION

    target_path = strategies_dir() / safe_name
    target_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "file_name": target_path.name,
        "file_path": target_path.as_posix(),
        "is_portfolio": is_portfolio,
        "symbol": data.get("symbol"),
        "formula": data.get("formula"),
        "formula_decoded": data.get("formula_decoded"),
        "best_score": data.get("best_score", 0),
        "vocab_version": data.get("vocab_version", ""),
        "data_file": data.get("data_file"),
        "timeframe": data.get("timeframe"),
    }
