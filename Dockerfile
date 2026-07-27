# ── Dockerfile — OKX AlphaPilot ─────────────────────────────────────
# 多阶段构建：PyTorch CPU 镜像 + 应用代码
# 默认 paper 模式，通过 .env / 环境变量切换 live
# ────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS builder

# 安装编译依赖（pyarrow/scipy 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# 先复制依赖文件，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── 运行阶段 ────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="OKX AlphaPilot"
LABEL description="OKX AlphaPilot | 量化研究与交易中枢 — 数据/训练/回测/实盘"
LABEL disclaimer="与 OKX 官方无隶属关系；默认 paper 模式，实盘需显式配置"

# 从 builder 复制已安装的 Python 包
COPY --from=builder /root/.local /root/.local

# 确保 user-local 包在路径中
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 复制应用代码
COPY . .

# 创建数据/策略/检查点目录（防止首次启动报错）
RUN mkdir -p data strategies checkpoints web/static/css web/static/js

# 暴露端口（与 .env 中 WEB_PORT 一致）
EXPOSE 8009

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8009/api/system', timeout=5)" || exit 1

# 默认启动命令
CMD ["python", "run.py"]
