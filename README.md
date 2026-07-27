# OKX AlphaPilot | 量化研究与交易中枢

> 从历史 K 线中自动挖掘可解释的「特征 + 算子」因子公式，转换为连续仓位信号，覆盖数据下载、模型训练、离线回测、实时分析和模拟/实盘交易的全链路量化平台。

**OKX AlphaPilot** 是一个面向 OKX 交易所的独立量化研究平台。它基于强化学习（REINFORCE）+ Looped Transformer 架构，自动搜索由 65 个因果特征和 66 个算子组成的最优因子公式，将因子值转换为 `[-1, 1]` 的连续仓位信号，多目标评分、Walk-Forward 验证等环节筛选出稳健的 Alpha 策略。

---

## ⚠️ 重要边界

| 边界 | 说明 |
|------|------|
| **默认模拟盘** | 默认模式是 `paper`，不会发送真实订单 |
| **本地训练** | 训练和回测只使用本地 Parquet，不调用 OKX 私有接口 |
| **显式实盘** | 真实交易必须显式开启 `live` 闸门，并配置完整 API 凭证 |
| **不保证收益** | 项目不保证收益，也不构成投资建议 |
| **分级执行** | 后续自动交易遵循 `backtest → paper → live` 分级 |

---

## ✨ 核心特性

### 因子挖掘引擎
- **强化学习搜索**：REINFORCE 策略梯度 + Actor-Critic baseline，搜索 8-token 公式序列
- **Looped Transformer**：GPT 风格模型 + QK-Norm 注意力 + 权重共享输出头
- **65 个因果特征**：趋势、波动率、反转、成交量、跨资产五类，严格防止未来函数泄露
- **66 个算子**：时序、截面、幅度变换、逻辑门，经"感染模型"校验防止因子退化为 Beta
- **多目标评分**：年化收益 + Sortino + Calmar + IC 稳定性 + 回撤控制 + Beta 中性
- **熵保护与重启**：自适应噪声 + 多级重启策略，应对策略坍塌
- **Elite Replay**：精英池记忆 + 衰减重放，加速收敛

### 策略融合与多因子组合
- **多因子策略生成**：支持选择多个独立的 Alpha 因子，按评分/IC 或等权重合成为组合策略
- **截面 Z-Score 标准化**：在合成时自动进行归一化，解决不同因子量级差异
- **全链条无缝兼容**：离线回测、实时分析与实盘交易无缝支持组合策略

### WebSocket 实时推送
- **行情与持仓推送**：集成 OKX v5 WebSocket (Public + Private)，实时推送 Ticker、账户余额、持仓及成交回报
- **自动断线重连**：内置指数退避重连与 Ping/Pong 20 秒心跳维持
- **前端状态灯指示**：实时呈现 WebSocket 推送状态 (🟢 WS 实时推送 / 🟡 REST 备用)

### 交易与风控
- **连续仓位信号**：Neutral Band + tanh 软压缩，输出 `[-1, 1]` 仓位
- **风控闸门**：杠杆上限、仓位上限、单日亏损、信号阈值、交易冷却检查
- **审计日志**：每笔交易决策全程记录（信号、风控、订单、模式）

### Web 平台
- **6 个页面**：总览、模型训练、策略回测、实时分析、实盘交易、数据管理
- **策略导入与组合**：支持一键导入外部/导出策略 JSON，可视化弹窗一键生成组合策略
- **实时图表**：Chart.js 资金曲线、训练曲线、价格信号图、仓位柱状图
- **深色量化主题**：现代深色 UI，专为长时间盯盘设计

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Web 前端 SPA                          │
│  总览 │ 模型训练 │ 策略回测 │ 实时分析 │ 实盘交易 │ 数据管理  │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP/REST
┌──────────────────────▼──────────────────────────────────┐
│              FastAPI 后端 (api/main.py)                   │
│  ┌─────────┬──────────┬──────────┬──────────┬────────┐  │
│  │  data   │ training │ backtest │ analysis │trading │  │
│  └────┬────┴────┬─────┴────┬─────┴────┬─────┴───┬────┘  │
│       │         │          │          │         │       │
│  ┌────▼────┐ ┌──▼───┐ ┌───▼────┐ ┌───▼────┐ ┌──▼────┐  │
│  │ Parquet │ │Engine│ │BT执行器│ │信号计算│ │交易执行│  │
│  │ 管理    │ │编排  │ │        │ │        │ │+审计  │  │
│  └────┬────┘ └──┬───┘ └───┬────┘ └───┬────┘ └──┬────┘  │
└───────┼─────────┼─────────┼──────────┼─────────┼───────┘
        │         │         │          │         │
┌───────▼─────────▼─────────▼──────────▼─────────▼───────┐
│                    核心层 (model/)                        │
│  AlphaGPT │ StackVM │ MT5Backtest │ Features │ Ops       │
│  REINFORCE + 熵保护 + Elite Replay + Walk-Forward        │
└─────────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────────┐
│              OKX v5 REST API (data_pipeline/)            │
│  K线下载 │ 行情 │ 品种发现 │ 下单 │ 持仓 │ 杠杆          │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 目录结构

```
okx-alpha-pilot/
├── config.py                       # 全局配置（路径、OKX API、风控）
├── run.py                          # 启动入口
├── .env.example                    # 环境变量模板
├── requirements.txt                # Python 依赖
│
├── model/                          # 机器学习核心
│   ├── alphagpt.py                 #   AlphaGPT 模型（QK-Norm + 权重共享）
│   ├── engine.py                   #   AlphaEngine 训练引擎（REINFORCE + 熵保护）
│   ├── vm.py                       #   StackVM 后缀表达式虚拟机
│   ├── backtest.py                 #   MT5Backtest 多目标评分
│   ├── features.py                 #   65 个因果特征工程
│   ├── ops.py                      #   66 个算子（时序/截面/幅度/逻辑）
│   ├── vocab.py                    #   词表自动派生 + 版本哈希
│   ├── registry.py                 #   声明式注册接口
│   ├── evaluator.py                #   因子评估器
│   ├── island_engine.py            #   多岛并行训练
│   └── config.py                   #   模型层配置（设备/训练参数/Reward）
│
├── strategy_manager/
│   └── signal.py                   # 因子 → 连续仓位转换（Neutral Band + tanh）
│
├── data_pipeline/                  # 数据管道
│   ├── okx_client.py               #   OKX v5 REST 客户端（公有+私有接口）
│   ├── parquet_manager.py          #   Parquet 读写管理
│   ├── timeframe_utils.py          #   K线周期工具
│   └── downloader.py               #   批量数据下载
│
├── api/                            # FastAPI 后端
│   ├── main.py                     #   应用入口 + 路由注册
│   ├── routers/                    #   路由层
│   │   ├── data.py                 #     数据管理（列表/下载/品种发现/删除）
│   │   ├── training.py             #     模型训练（启动/状态/曲线/断点续训）
│   │   ├── backtest.py             #     策略回测（资金曲线/绩效指标）
│   │   ├── analysis.py             #     实时分析（OKX/MT5/TradingView 信号）
│   │   └── trading.py              #     实盘交易（paper/live + 审计日志）
│   └── services/                   #   服务层
│       ├── training_service.py     #     训练编排（后台线程 + 实时进度）
│       ├── backtest_service.py     #     回测执行（PnL/Sortino/Calmar/回撤）
│       ├── analysis_service.py     #     信号计算（实时K线 → 因子 → 仓位）
│       ├── trading_service.py      #     交易执行（风控 + 审计日志）
│       └── strategy_service.py     #     策略加载/解码/保存
│
├── web/                            # 前端 SPA
│   ├── index.html                  #   6 页面单页应用
│   └── static/
│       ├── css/style.css           #   现代深色量化主题
│       └── js/app.js               #   SPA 路由 + Chart.js 图表
│
├── data/                           # Parquet 数据文件（自动创建）
├── strategies/                     # 策略 JSON 文件（自动创建）
└── checkpoints/                    # 训练检查点（自动创建）
```

---
## 准备开始-注册获取 OKX API

1. **注册 OKX**：[点击okx官网注册,佣金享5%优惠](https://www.gtohfmmy.com/join/6746503)
   - 使用上面邀请码注册并完成任务，最高获 100 USDT 奖励，交易佣金优惠 5%。具体奖励、地区限制和活动规则以 OKX 页面显示为准。
2. 登录 OKX，点击右上角个人中心，进入“API 管理”，创建 API。
3. API 权限至少需要“读取”；需要下单时增加“交易”权限。
4. **不要授予提现权限**。建议设置服务器公网 IP 白名单。
5. 妥善保存 `API Key`、`Secret Key` 和创建时填写的 `Passphrase`，关闭页面后部分信息可能无法再次查看。
6. 资金需要划转到交易账户后才能用于交易。模拟盘和实盘应分别创建对应环境的 API Key。

<img width="1635" height="795" alt="OKX API 创建示意图" src="https://github.com/user-attachments/assets/66a8685f-e428-4fb8-afc9-efde9053a223" />

## 🚀 快速开始

### 1. 安装依赖

```bash
cd okx-alpha-pilot
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`：

```ini
# 交易模式：paper（默认模拟盘）或 live（实盘）
TRADING_MODE=paper

# OKX API 凭证（实盘需要，模拟盘可留空）
OKX_API_KEY=
OKX_API_SECRET=
OKX_API_PASSPHRASE=
OKX_API_BASE=https://www.okx.com
OKX_SIMULATED=1

# Web 服务
WEB_HOST=0.0.0.0
WEB_PORT=8009
```

### 3. 启动平台

```bash
python run.py
```

或指定端口 / 开发模式：

```bash
python run.py --port 9000
python run.py --reload
```

启动后访问 **http://localhost:8009**。

### 4. Docker 部署（推荐）

#### 方式 A：docker compose（推荐）

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 填写 OKX 凭证（实盘需要）

# 2. 构建并启动
docker compose up -d --build

# 3. 查看日志
docker compose logs -f

# 4. 停止
docker compose down
```

访问 **http://localhost:8009**。

#### 方式 B：docker 命令

```bash
# 构建镜像
docker build -t okx-alphapilot .

# 启动容器（挂载数据目录持久化）
docker run -d \
  --name alphapilot \
  --restart unless-stopped \
  -p 8009:8009 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/strategies:/app/strategies \
  -v $(pwd)/checkpoints:/app/checkpoints \
  okx-alphapilot

# 查看日志
docker logs -f alphapilot

# 停止删除
docker stop alphapilot && docker rm alphapilot
```

#### Docker 部署说明

| 项目 | 说明 |
|------|------|
| 端口 | 默认 8009，可通过 `.env` 中 `WEB_PORT` 修改 |
| 数据持久化 | `data/` `strategies/` `checkpoints/` 通过 volume 挂载，容器重建后不丢失 |
| 默认模式 | `paper`（模拟盘），设置 `TRADING_MODE=live` + 凭证后切换实盘 |
| 资源限制 | 默认 4G 内存 / 2 核 CPU，可在 `docker-compose.yml` 中调整 |
| 健康检查 | 每 30 秒检查 `/api/system` 接口 |
| 日志 | JSON 格式，单文件 10MB，最多 3 个 |

### 5. 典型工作流

```
数据管理          模型训练          策略回测          实时分析          实盘交易
   │                │                │                │                │
   ▼                ▼                ▼                ▼                ▼
下载 K线  ──→  选择 Parquet   ──→  选择策略     ──→  选择数据源   ──→  选择策略
BTC-USDT       训练 9000 步        选择数据          OKX 实时          设置本金
 1H 周期        查看训练曲线        设置手续费        计算信号          设置杠杆
 2000 根        导出策略            查看资金曲线      查看仓位          执行信号
                                                                  查看审计日志
```

---

## 📊 功能页面

### 1. 总览
- 系统状态：交易模式、策略数量、数据文件数、词表大小
- 最近策略列表和数据文件列表
- 系统合规声明与免责声明

### 2. 模型训练
- **选择 Parquet**：从本地数据文件中选择训练数据
- **断点续训**：从已有检查点恢复训练
- **奖励模式**：FTMO（年化优先）/ Standard（平衡）/ Forex（均值回归）
- **训练曲线**：最优分数、平均奖励、策略熵的实时图表
- **导出策略**：训练完成后自动保存策略 JSON

| 参数 | 默认值 | 说明 |
|------|--------|------|
| BATCH_SIZE | 192 | 每步采样公式数 |
| TRAIN_STEPS | 9000 | 训练步数 |
| MAX_FORMULA_LEN | 8 | 公式长度上限 |
| ELITE_POOL_SIZE | 60 | 精英池大小 |
| MAX_RESTARTS | 55 | 最大重启次数 |

### 3. 策略回测
- **选择策略和数据**：指定策略 JSON 和 Parquet 数据文件
- **交易成本**：可调手续费率和滑点
- **资金曲线**：Chart.js 绘制的权益变化图
- **绩效指标**：

| 指标 | 说明 |
|------|------|
| 总收益 | 回测期间总收益率 |
| 年化收益 | 年化对数收益 |
| 最大回撤 | 峰值到谷值的最大跌幅 |
| 夏普比率 | 风险调整后收益 |
| Sortino | 下行风险调整后收益 |
| Calmar | 年化收益 / 最大回撤 |
| 胜率 | 盈利 K 线占比 |
| 综合评分 | 多目标加权评分 |

### 4. 实时分析
- **数据源选择**：
  - **OKX 实时**：从 OKX API 获取最新 K 线
  - **本地 Parquet**：MT5 / TradingView 导出的本地数据
- **信号计算**：收盘后更新因子值和目标仓位
- **可视化**：价格 + 因子值双轴图、仓位柱状图
- **信号统计**：做多/做空/空仓 K 线数及占比

### 5. 实盘交易
- **双模式**：paper（默认模拟盘）/ live（显式开启）
- **风控闸门**：
  - 杠杆上限检查（默认 max 20x）
  - 仓位占比上限（默认 30%）
  - 信号阈值检查（|signal| ≥ 0.05）
  - 信号范围校验（[-1, 1]）
- **审计日志**：记录每笔交易的信号、风控结果、订单详情

### 6. 数据管理
- **下载 K 线**：从 OKX 下载指定品种和周期的历史数据
- **品种发现**：自动发现 OKX 可用 SWAP 合约（加密、贵金属、指数等）
- **文件管理**：查看、删除本地 Parquet 文件

---

## 🔌 API 文档

启动后访问交互式 API 文档：
- **Swagger UI**：http://localhost:8009/docs
- **ReDoc**：http://localhost:8009/redoc

### 核心端点

| 方法 | 路径 | 功能 |
|------|------|------|
| `GET` | `/api/system` | 系统状态 |
| `GET` | `/api/data/parquets` | 列出 Parquet 文件 |
| `POST` | `/api/data/download` | 下载 K 线数据 |
| `GET` | `/api/data/instruments` | 发现 OKX 品种 |
| `POST` | `/api/training/start` | 启动训练 |
| `GET` | `/api/training/status` | 训练状态 |
| `GET` | `/api/training/history` | 训练曲线数据 |
| `GET` | `/api/training/checkpoints` | 列出检查点 |
| `GET` | `/api/training/strategies` | 列出已保存策略 |
| `POST` | `/api/backtest/run` | 执行回测 |
| `POST` | `/api/analysis/okx` | OKX 实时分析 |
| `POST` | `/api/analysis/parquet` | 本地数据分析 |
| `GET` | `/api/trading/status` | 交易服务状态 |
| `POST` | `/api/trading/execute` | 执行交易信号 |
| `POST` | `/api/trading/close/{inst_id}` | 平仓 |
| `GET` | `/api/trading/audit` | 审计日志 |
| `GET` | `/api/trading/config` | 交易配置 |

---

## ⚙️ 配置详解

### 全局配置 (`config.py`)

| 配置项 | 默认值 | 环境变量 | 说明 |
|--------|--------|----------|------|
| `TRADING_MODE` | `paper` | `TRADING_MODE` | 交易模式：paper / live |
| `OKX_API_KEY` | `""` | `OKX_API_KEY` | OKX API Key |
| `OKX_API_SECRET` | `""` | `OKX_API_SECRET` | OKX API Secret |
| `OKX_API_PASSPHRASE` | `""` | `OKX_API_PASSPHRASE` | OKX Passphrase |
| `OKX_API_BASE` | `https://www.okx.com` | `OKX_API_BASE` | OKX API 基础 URL |
| `OKX_API_SIMULATED` | `True` | `OKX_SIMULATED` | 是否模拟盘 |
| `OKX_BROKER_TAG` | `c314b0aecb5bBCDE` | `OKX_BROKER_TAG` | 经纪商标识 |
| `DEFAULT_CAPITAL` | `10000.0` | — | 默认本金 (USDT) |
| `DEFAULT_LEVERAGE` | `5` | — | 默认杠杆 |
| `MAX_LEVERAGE` | `20` | — | 最大杠杆 |
| `MAX_DAILY_LOSS_PCT` | `0.10` | — | 单日最大亏损 10% |
| `MAX_POSITION_PCT` | `0.30` | — | 单品种最大仓位占比 |
| `WEB_HOST` | `0.0.0.0` | `WEB_HOST` | Web 监听地址 |
| `WEB_PORT` | `8009` | `WEB_PORT` | Web 监听端口 |

### 模型配置 (`model/config.py`)

模型配置在 `model/config.py` 的 `ModelConfig` 类中，包含：
- 训练设备（强制 CPU，因小张量场景 CPU 反而更快）
- 训练参数（BATCH_SIZE、TRAIN_STEPS、MAX_FORMULA_LEN）
- Reward 权重（FTMO / Standard / Forex 三种模式）
- 熵保护参数（自适应噪声 + 重启策略）
- Elite Replay 配置
- Walk-Forward 折叠参数

---

## 📈 适配品种

项目根据 OKX 区域 API 返回的 instrument 自动发现品种，例如：

| 类型 | 示例 |
|------|------|
| 加密货币 | `BTC-USDT-SWAP`、`ETH-USDT-SWAP` |
| 贵金属 | `XAU-USDT-SWAP` |
| 指数 | `SPX-USDT-SWAP` |
| 股票 | `AAPL-USDT-SWAP` |

> TradFi instrument 的可用性取决于 OKX 区域、账户权限和接口实时返回结果。

---

## 🔒 免责声明

- 本项目与 **OKX 官方无隶属或背书关系**；名称中的 OKX 仅表示主要适配的交易所接口
- 项目**不保证收益，也不构成投资建议**
- 加密货币交易具有高风险，可能导致全部本金损失
- 使用实盘交易功能前，请充分了解风险并完成模拟盘验证
- 用户应遵守所在地区的法律法规

---

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| 机器学习 | PyTorch、REINFORCE、Looped Transformer |
| 后端 | FastAPI、Uvicorn、Pydantic |
| 数据 | Pandas、PyArrow、Parquet |
| 前端 | 原生 JS、Chart.js、CSS3 |
| API | OKX v5 REST API |
| 认证 | HMAC-SHA256 签名 |

---

## 📝 开发指南

### 添加新特征

在 `model/features.py` 中通过 `FeatureSpec` 注册：

```python
FEATURE_REGISTRY.register(FeatureSpec(
    name="MY_FEATURE",
    category="custom",
    compute=lambda raw_dict: my_compute_function(raw_dict),
))
```

### 添加新算子

在 `model/ops.py` 中注册：

```python
@OPS_REGISTRY.register("MY_OP", arity=1)
def my_op(x):
    return torch.sigmoid(x)
```

### 添加新 API 端点

1. 在 `api/routers/` 创建路由文件
2. 在 `api/services/` 创建服务文件
3. 在 `api/main.py` 中注册路由

### 开发模式

```bash
python run.py --reload
```

---

## ❓ 常见问题

<details>
<summary><b>Q: 为什么训练强制用 CPU 而不是 GPU？</b></summary>

本项目的张量很小（单品种 × 3500 K线 × 30 特征），单个算子的计算量小于 CUDA kernel 启动开销。实测 CPU 比 GPU 快约 2.3 倍（1.91s/步 vs 4.48s/步）。若后续改为批量并行公式评估，可切回 GPU。
</details>

<details>
<summary><b>Q: 如何启用实盘交易？</b></summary>

1. 在 `.env` 中设置 `TRADING_MODE=live`
2. 配置完整的 OKX API 凭证（Key、Secret、Passphrase）
3. 设置 `OKX_SIMULATED=0`（关闭模拟盘）
4. 重启服务
5. 系统会在每笔订单前进行风控检查，并在审计日志中记录

**警告**：实盘交易会发送真实订单，可能导致真实资金损失。
</details>

<details>
<summary><b>Q: 如何移除经纪商标识？</b></summary>

在 `.env` 中设置：
```ini
OKX_BROKER_TAG=
```
这会清空经纪商标识，订单中将不再携带此标签。
</details>

<details>
<summary><b>Q: 训练中断后如何续训？</b></summary>

1. 在「模型训练」页面选择已有检查点
2. 点击「启动训练」
3. 系统会从检查点恢复模型权重、优化器状态和训练历史
</details>

<details>
<summary><b>Q: 支持哪些 K 线周期？</b></summary>

OKX 支持的所有周期：`1m`、`3m`、`5m`、`15m`、`30m`、`1H`、`2H`、`4H`、`6H`、`12H`、`1D`、`1W`、`1M`。
</details>

---

## 📄 License

本项目仅供学习和研究用途。使用者需遵守所在地区的法律法规，自行承担使用风险。

---

**与 OKX 官方无隶属或背标关系 · 不构成投资建议 · 加密货币交易高风险**
