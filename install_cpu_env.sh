#!/bin/bash
# install_cpu_env.sh - 全新环境一键安装纯 CPU 依赖（杜绝 CUDA 显卡冗余）
set -e

echo "=== 正在从 PyTorch 官方源安装纯 CPU 版本 ==="
pip install "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cpu

echo "=== 正在安装其余项目运行依赖 ==="
pip install -r requirements.txt

echo "=== 安装验证 ==="
python -c "import torch; print(f'PyTorch 成功安装: 版本 {torch.__version__}, CUDA 显卡依赖: {torch.cuda.is_available()}')"
