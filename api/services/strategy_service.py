"""
api/services/strategy_service.py — 策略管理服务

加载、保存、解码策略 JSON 文件。
"""
import json
import pathlib
from typing import Optional

from config import Config
from model.vocab import FORMULA_VOCAB, VOCAB_VERSION


def strategies_dir() -> pathlib.Path:
    d = pathlib.Path(Config.STRATEGIES_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_strategies() -> list[dict]:
    """列出所有已保存的策略。"""
    files = sorted(strategies_dir().glob("*.json"))
    result = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            result.append({
                "file_name": f.name,
                "file_path": f.as_posix(),  # 用正斜杠，避免 Windows 反斜杠在 JS 中被转义
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
    """加载策略 JSON。"""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"策略文件不存在: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if "formula_decoded" not in data:
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
    """保存策略到 JSON。

    文件名格式：best_{symbol}_{timeframe}.json（有时间周期时），
                best_{symbol}.json（无时间周期时）。
    """
    parts = []
    if symbol:
        parts.append(symbol)
    if timeframe:
        # 清理非法字符
        tf_safe = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(timeframe))
        parts.append(tf_safe)
    tag = "_".join(parts) if parts else ""
    path = strategies_dir() / (f"best_{tag}.json" if tag else "best.json")
    data = {
        "vocab_version": VOCAB_VERSION,
        "symbol": symbol,
        "timeframe": timeframe,
        "data_file": data_file,
        "formula": formula,
        "formula_decoded": decode_formula(formula),
        "best_score": score,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def delete_strategy(path: str) -> dict:
    """删除策略 JSON 文件。

    安全校验：文件必须位于 strategies 目录内，防止路径遍历攻击。

    Returns:
        {"ok": True, "deleted": file_name} 或抛出异常
    """
    p = pathlib.Path(path).resolve()
    base = strategies_dir().resolve()

    # 路径安全校验：确保文件在 strategies 目录内
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
    """导入外部策略 JSON 文件。

    Args:
        content_bytes: 上传的文件内容字节
        filename: 原始文件名

    Returns:
        导入成功后的策略元数据字典
    """
    if not content_bytes:
        raise ValueError("上传的文件内容为空")

    try:
        data = json.loads(content_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"无效的 JSON 格式: {e}")

    if not isinstance(data, dict):
        raise ValueError("策略 JSON 必须为对象格式")

    formula = data.get("formula")
    if not formula or not isinstance(formula, list):
        raise ValueError("策略文件缺少有效的 'formula' 算子序列")

    # 路径安全处理
    p_name = pathlib.Path(filename).name
    safe_name = "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in p_name)
    if not safe_name.endswith(".json"):
        safe_name += ".json"

    # 补充解码公式和版本
    if "formula_decoded" not in data:
        data["formula_decoded"] = decode_formula(formula)
    if "vocab_version" not in data:
        data["vocab_version"] = VOCAB_VERSION

    target_path = strategies_dir() / safe_name
    target_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "file_name": target_path.name,
        "file_path": target_path.as_posix(),
        "symbol": data.get("symbol"),
        "formula": formula,
        "formula_decoded": data.get("formula_decoded"),
        "best_score": data.get("best_score", 0),
        "vocab_version": data.get("vocab_version", ""),
        "data_file": data.get("data_file"),
        "timeframe": data.get("timeframe"),
    }

