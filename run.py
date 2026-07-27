#!/usr/bin/env python3
"""
run.py — OKX AlphaPilot 启动入口

用法:
  python run.py              # 启动 Web 服务（默认 0.0.0.0:8009）
  python run.py --port 9000  # 指定端口
  python run.py --reload     # 开发模式热重载

环境变量:
  TRADING_MODE       paper / live（默认 paper）
  OKX_API_KEY        OKX API Key（实盘需要）
  OKX_API_SECRET     OKX API Secret（实盘需要）
  OKX_API_PASSPHRASE OKX Passphrase（实盘需要）
  OKX_SIMULATED      1=模拟盘 0=实盘（默认 1）
  WEB_HOST           监听地址（默认 0.0.0.0）
  WEB_PORT           监听端口（默认 8009）
"""
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="OKX AlphaPilot 量化研究与交易中枢")
    parser.add_argument("--host", default=None, help="监听地址")
    parser.add_argument("--port", type=int, default=None, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    from config import Config

    host = args.host or Config.WEB_HOST
    port = args.port or Config.WEB_PORT

    print(f"\n{'='*60}")
    print(f"  OKX AlphaPilot | 量化研究与交易中枢")
    print(f"{'='*60}")
    print(f"  启动地址: http://{host}:{port}")
    print(f"  交易模式: {Config.TRADING_MODE} ({'实盘' if Config.is_live() else '模拟'})")
    print(f"{'='*60}\n")

    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
