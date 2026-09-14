# OKX AlphaPilot | 量化研究与自进化交易中枢

> 从历史 K 线中自动挖掘可解释的「特征 + 算子」因子公式，并结合 **Reef 持续自进化闭环**（实盘因果归因、微观体制感知、自适应风控、痛点驱动遗传繁衍与零停机原子交付），覆盖数据下载、离线挖掘、回测验证、实盘交易与在线持续进化的全链路量化平台。

**OKX AlphaPilot** 是一个面向 OKX 交易所的独立量化研究与自进化交易平台。它基于强化学习（REINFORCE）+ Looped Transformer 架构，自动搜索由 65 个因果特征和 66 个算子组成的最优因子公式，结合微观波动率体制感知（Harness）与持续在线进化中枢（Reef），在动态多变的市场中自适应抵御震荡回撤并持续迭代 Alpha 策略。

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

### 🧬 Reef 持续自进化闭环 (Continual Self-Improvement)
- **Phase 1: 真实对齐与因果归因 (Observe & Credit Alignment)**：
  - **决策凭证追踪**：每次信号生成与发单分配全局唯一 `receipt_id`，快照记录价格、信号强度与风控状态。
  - **OKX 账单与持仓对齐**：自动同步 OKX 历史平仓记录 (`positions-history`) 与账单流水 (`bills`)，精准还原真实成交收益、滑点摩擦、手续费与资金费率。
  - **因果轨迹沉淀**：建立本地高置信度实盘因果轨迹库 (`data/evolution/trajectories.jsonl`)，为在线优化提供带标签的实盘数据集。
- **Phase 2: 微观体制感知与动态调谐 (Adaptive Harness)**：
  - **波动率体制感知 (`MarketRegimeDetector`)**：通过短期 ATR 与基线 ATR 比率及动量斜率，实时识别 `VOL_EXPANSION`（波动扩张）、`VOL_COMPRESSION`（波动压缩/窄幅阴跌）与 `NORMAL`（常态）。
  - **动态死区缓冲 (Dynamic Neutral Band)**：阴跌缩量期动态收窄死区敏捷出场；单边行情放宽死区避免震荡出局。
  - **自适应止损点 (Dynamic SL)**：低波阴跌期自适应收紧止损至 1.5%~2.0%，防范钝刀割肉；单边扩张期适度放宽至 3.5%~4.0%，容忍正常技术回踩（实盘回测验证减少 42.7% 回撤）。
  - **离线回放调优 (`HarnessOptimizer`)**：基于真实实盘轨迹自动网格调谐最优参数，热更新生效至 `active_harness.json`。
- **Phase 3: 痛点进化、影子门禁与原子交付 (Grow, Shadow Pool & Commit)**：
  - **痛点引导遗传繁衍 (`GrowEngine`)**：分析历史实盘失败与盈利轨迹，对假突破反转、慢阴跌等痛点施加定向适应度偏置（痛点惩罚与顺势奖励），后台异步变异繁衍策略公式。
  - **5 重严苛交付门禁 (`ShadowEvaluator`)**：进入影子池候选策略必须全部通过：
    1. *因果无未来函数门禁 (No Lookahead)*：逐 bar 递增计算保证严格无时序未来泄露。
    2. *非退化方差门禁 (Non-Degenerate)*：因子截面标准差 $\ge 1\times 10^{-4}$，杜绝全 0/常量退化因子。
    3. *样本外得分超越门禁 (Score Improvement)*：综合多目标评分超越当前实盘基准策略至少 $+3\%$。
    4. *最大回撤防御门禁 (Drawdown Guard)*：最大回撤不超过基准策略的 1.05 倍，确保风险受控。
    5. *逆波兰栈机实盘兼容门禁 (StackVM Compatibility)*：全生命周期执行零 NaN、零 Inf、零崩溃。
  - **零停机原子交付与归档 (`CommitManager`)**：
    - 自动将当前实盘策略备份归档至 `strategies/archive/`，支持秒级无损回滚。
    - 原子替换策略文件，实盘交易循环下一 Tick 自动热加载生效，交易零停机中断。
    - 全生命周期交付日志持久化至 `data/evolution/commits.jsonl`。

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
- **连续仓位信号**：动态 Neutral Band + tanh 软压缩，输出 `[-1, 1]` 仓位
- **风控闸门**：动态止损保护、杠杆上限、仓位上限、单日亏损、信号阈值、交易冷却检查
- **审计日志**：每笔交易决策全程记录（信号、风控、凭证 `receipt_id`、订单、模式）

### Web 平台
- **7 个页面**：总览、模型训练、策略回测、实时分析、实盘交易、Reef 自进化、数据管理
- **策略导入与组合**：支持一键导入外部/导出策略 JSON，可视化弹窗一键生成组合策略
- **实时图表**：Chart.js 资金曲线、训练曲线、价格信号图、仓位柱状图、自进化态势感知
- **深色量化主题**：现代深色 UI，专为长时间盯盘设计

---

## 🏗️ 系统架构

```
┌────────────────────────────────────────────────────────────────────────┐
│                              Web 前端 SPA                               │
│  总览 │ 模型训练 │ 策略回测 │ 实时分析 │ 实盘交易 │ Reef 自进化 │ 数据管理   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP/REST
┌───────────────────────────────────▼────────────────────────────────────┐
│                       FastAPI 后端 (api/main.py)                        │
│  ┌─────────┬──────────┬──────────┬──────────┬──────────┬────────────┐  │
│  │  data   │ training │ backtest │ analysis │ trading  │ evolution  │  │
│  └────┬────┴────┬─────┴────┬─────┴────┬─────┴───┬──────┴─────┬──────┘  │
│       │         │          │          │         │            │         │
│  ┌────▼────┐ ┌──▼───┐ ┌───▼────┐ ┌───▼────┐ ┌──▼────┐   ┌────▼───────┐ │
│  │ Parquet │ │Engine│ │BT执行器│ │信号计算│ │交易执行│   │ Reef进化管线│ │
│  │ 管理    │ │编排  │ │        │ │        │ │+审计  │   │            │ │
│  └────┬────┘ └──┬───┘ └───┬────┘ └───┬────┘ └──┬────┘   └────┬───────┘ │
└───────┼─────────┼─────────┼──────────┼─────────┼──────────────┼────────┘
        │         │         │          │         │              │
┌───────▼─────────▼─────────▼──────────▼─────────▼──────────────▼────────┐
│                        持续自进化与核心层                                │
│  AlphaGPT │ StackVM │ MT5Backtest │ Features │ Ops                     │
│  ────────────── Reef Continual Self-Improving Loop ──────────────────  │
│  [Observe 账单对齐] → [Adaptive Harness 动态风控] → [Grow 痛点遗传变异]   │
│                 → [Shadow 5重门禁] → [Commit 原子热交付]               │
└──────────────────────────────────────┬─────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼─────────────────────────────────┐
│                    OKX v5 REST / WebSocket (data_pipeline/)            │
│  行情与K线 │ 账户资产 │ 仓位流水 (positions-history) │ 账单 (bills) │ 交易下单   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 目录结构

```
okx-alpha-pilot/
├── config.py                       # 全局配置（路径、OKX API、风控、Reef进化参数）
├── run.py                          # 启动入口
├── .env.example                    # 环境变量模板
├── requirements.txt                # Python 依赖
│
├── evolution/                      # 🧬 Reef 持续自进化中枢
│   ├── __init__.py                 #   统一导出接口
│   ├── observer.py                 #   Phase 1: 真实对齐与账单归因 (ObserveEngine)
│   ├── harness.py                  #   Phase 2: 微观态势感知与自适应风控 (AdaptiveHarnessPolicy)
│   ├── grow.py                     #   Phase 3: 痛点变异进化引擎 (GrowEngine)
│   ├── shadow.py                   #   Phase 3: 5 重影子门禁评估器 (ShadowEvaluator)
│   └── commit.py                   #   Phase 3: 原子热替换与归档管理器 (CommitManager)
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
│   └── signal.py                   # 因子 → 连续仓位转换（动态 Neutral Band + tanh）
│
├── data_pipeline/                  # 数据管道
│   ├── okx_client.py               #   OKX v5 REST 客户端（公有+私有接口/账单与历史持仓）
│   ├── okx_ws_client.py            #   OKX v5 WebSocket 实时订阅客户端
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
│   │   ├── trading.py              #     实盘交易（paper/live + 审计日志）
│   │   ├── portfolio.py            #     组合策略（多因子融合与构建）
│   │   └── evolution.py            #     Reef 自进化（状态/轨迹/调谐/Grow/影子/Commit）
│   └── services/                   #   服务层
│       ├── training_service.py     #     训练编排（后台线程 + 实时进度）
│       ├── backtest_service.py     #     回测执行（PnL/Sortino/Calmar/回撤）
│       ├── analysis_service.py     #     信号计算（实时K线 → 因子 → 仓位）
│       ├── trading_service.py      #     交易执行（风控 + 动态止损 + 审计日志）
│       └── strategy_service.py     #     策略加载/解码/保存与多因子组合
│
├── web/                            # 前端 SPA
│   ├── index.html                  #   7 页面单页应用（包含 Reef 自进化中枢）
│   └── static/
│       ├── css/style.css           #   现代深色量化主题
│       └── js/app.js               #   SPA 路由 + Chart.js 图表 + 自进化操作面板
│
├── data/                           # 数据持久化目录
│   └── evolution/                  #   Reef 进化数据（trajectories/active_harness/shadow_pool/commits）
├── strategies/                     # 策略 JSON 文件
│   └── archive/                    #   热交付历史备份归档
└── checkpoints/                    # 训练检查点
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
数据管理         模型训练         策略回测         实盘交易           Reef 持续自进化中枢
   │               │               │               │                       │
   ▼               ▼               ▼               ▼                       ▼
下载 K线 ───→ 选择 Parquet ───→ 选择策略 ───→ 执行信号 ───→ [Observe] 同步 OKX 账单与对齐轨迹
ETH-USDT      训练 9000 步       离线回测        动态止损               │
 ≥15m 周期     导出初始策略       多指标评估      凭证审计          [Adaptive Harness]
 ≥4000 根                                                    识别波动率体制 & 动态死区
                                                                        │
                                                             [Grow] 痛点引导变异进化
                                                                        │
                                                             [Shadow] 5重安全门禁检验
                                                                        │
                                                             [Commit] 零停机原子交付实盘
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

> 💡 **训练数据推荐配置（最佳实践）**：
> - **推荐 K 线周期**：建议选择 **15m 周期以上（如 15m、30m、1H）** 进行训练。超短周期（如 1m/3m/5m）市场微观噪声大且换手极频，极易被交易手续费和滑点损耗侵蚀；15m 以上周期信噪比显著提升，波段趋势性更强。
> - **推荐 K 线数量**：建议使用 **4000 根以上** 的历史 K 线（如 4000 根 15m 覆盖约 41 天，4000 根 1H 覆盖约 166 天）。充足的样本量不仅能让 500 周期的滚动特征归一化充分收敛，还能跨越不同市场行情状态，有效防止样本内过拟合。

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
  - 动态自适应止损点（结合 Harness 微观体制自适应收紧/放宽）
  - 杠杆上限检查（默认 max 20x）
  - 仓位占比上限（默认 30%）
  - 信号阈值检查（|signal| ≥ 0.05）
  - 信号范围校验（[-1, 1]）
- **审计日志**：记录每笔交易的信号、风控结果、订单详情以及全局唯一的决策凭证 `receipt_id`

### 6. Reef 自进化操作中枢
- **自进化全景总览**：一屏掌握已对齐实盘轨迹数、实盘胜率、累计净盈亏，实时监控 Observe、Harness、Grow、Shadow 与 Commit 五大模块运行状态。
- **微观体制实时监控**：
  - 实时检测当前市场波动率体制（`VOL_EXPANSION` 扩张 / `VOL_COMPRESSION` 压缩 / `NORMAL` 平衡）。
  - 动态呈现当前周期的 ATR14 比率、动态 Neutral Band 死区范围（如 `[-0.15, 0.15]`）与自适应止损线（如 `1.5%`）。
- **一键账单对齐与回放寻优**：
  - 点击「同步 OKX 真实账单」自动拉取最新成交平仓单，精准溯源计算真实滑点与手续费，生成高置信度轨迹。
  - 点击「寻优自适应配置」基于沉淀的真实实盘轨迹回放微观风控参数，沉淀至 `active_harness.json`。
- **Grow 策略进化交互**：
  - 支持自定义进化代数（步数，如 100 步）。
  - 点击「启动 Grow 策略进化」由后台守护线程在不影响实盘的前提下进行变异繁衍。
  - 前端支持毫秒级轮询：显示「● Grow 进化运行中 (Step X/Y, Best Score: Z)」，并在完成后自动刷新影子池。
- **影子观察池与 5 重门禁透视**：
  - 呈现进入候选池的新策略代码、Token 公式逆波兰表达式、验证集评分对比与夏普提升幅度。
  - 详细列示 5 重门禁（No-Lookahead、方差非退化、超越基准 $\ge 3\%$、回撤不劣化、StackVM 兼容性）的通过状态（✅ PASSED）。
- **一键原子热交付 (Commit)**：
  - 对已通过全部门禁的胜出候选策略，支持管理员点击「确认交付实盘 (Commit)」。
  - 系统原子化替换活跃实盘策略，自动备份归档老版本至 `strategies/archive/`，实时交易服务下一 Tick 自动热重载，交易全程零停机。

### 7. 数据管理
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

### Reef 自进化端点 (`/api/evolution/*`)

| 方法 | 路径 | 功能 |
|------|------|------|
| `GET` | `/api/evolution/status` | 获取自进化流水线（五大模块）全景状态 |
| `GET` | `/api/evolution/trajectories` | 查询实盘因果对齐轨迹列表与明细 |
| `GET` | `/api/evolution/stats` | 查询实盘交易归因统计指标（胜率、净收益等） |
| `POST` | `/api/evolution/sync` | 触发 OKX 账单与持仓对齐同步（`lookback_days`） |
| `GET` | `/api/evolution/harness` | 获取自适应风控配置与实时体制参数 |
| `POST` | `/api/evolution/harness/tune` | 触发基于实盘轨迹的 Harness 离线寻优 |
| `POST` | `/api/evolution/grow/start` | 启动后台策略遗传变异进化任务 |
| `GET` | `/api/evolution/grow/status` | 查询策略进化任务当前进度与最佳候选 |
| `POST` | `/api/evolution/grow/stop` | 终止正在运行的策略进化任务 |
| `GET` | `/api/evolution/shadow` | 查询影子候选池与 5 重门禁评估报告 |
| `POST` | `/api/evolution/commit` | 触发影子策略原子热替换交付（自动归档） |
| `GET` | `/api/evolution/commits` | 查询策略交付与归档审计历史 |

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
| `DEFAULT_CAPITAL` | `10000.0` | — | 默认本金 (USDT) |
| `DEFAULT_LEVERAGE` | `5` | — | 默认杠杆 |
| `MAX_LEVERAGE` | `20` | — | 最大杠杆 |
| `MAX_DAILY_LOSS_PCT` | `0.10` | — | 单日最大亏损 10% |
| `MAX_POSITION_PCT` | `0.30` | — | 单品种最大仓位占比 |
| `WEB_HOST` | `0.0.0.0` | `WEB_HOST` | Web 监听地址 |
| `WEB_PORT` | `8009` | `WEB_PORT` | Web 监听端口 |

### 自进化配置 (`config.py`)

| 配置项 | 默认值 | 环境变量 | 说明 |
|--------|--------|----------|------|
| `ENABLE_DYNAMIC_HARNESS` | `True` | `ENABLE_DYNAMIC_HARNESS` | 是否开启自适应微观体制感知与动态 Neutral Band / 动态止损 |
| `HARNESS_MIN_SL` | `0.015` | `HARNESS_MIN_SL` | 动态止损下限 (1.5%，低波阴跌保护) |
| `HARNESS_MAX_SL` | `0.040` | `HARNESS_MAX_SL` | 动态止损上限 (4.0%，高波扩张容忍度) |
| `HARNESS_DEFAULT_SL` | `0.030` | `HARNESS_DEFAULT_SL` | 默认基准止损线 (3.0%) |
| `AUTO_COMMIT_STRATEGY` | `False` | `AUTO_COMMIT_STRATEGY` | 5 重门禁全过时是否自动执行原子替换（默认 False，由人工确认） |
| `SHADOW_MIN_IMPROVEMENT`| `0.03` | `SHADOW_MIN_IMPROVEMENT` | 候选策略超越基准策略的最小评分提升幅度 (3%) |
| `EVOLUTION_WORKERS` | `1` | `EVOLUTION_WORKERS` | 后台进化工作线程数 |

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
<summary><b>Q: 训练中断后如何续训？</b></summary>

1. 在「模型训练」页面选择已有检查点
2. 点击「启动训练」
3. 系统会从检查点恢复模型权重、优化器状态和训练历史
</details>

<details>
<summary><b>Q: 训练推荐使用什么 K 线周期和数据量？</b></summary>

强烈推荐使用 **15m 周期以上（15m / 30m / 1H）**，并下载 **4000 根 K 线** 进行训练：
1. **抵御手续费与滑点**：1m/3m/5m 微观高频噪点多，回测扣除 0.05% 手续费后容易大幅回吐；15m 以上周期单笔盈利空间更大，能有效覆盖交易成本。
2. **特征稳定收敛**：系统的 65 个因果特征依赖滚动 500 周期标准化，4000 根 K 线能保证特征充分预热、样本充分，避免短序列过拟合。
</details>

<details>
<summary><b>Q: 支持哪些 K 线周期？</b></summary>

OKX 支持的所有周期：`1m`、`3m`、`5m`、`15m`、`30m`、`1H`、`2H`、`4H`、`6H`、`12H`、`1D`、`1W`、`1M`。
</details>

<details>
<summary><b>Q: Reef 持续自进化系统是如何保证安全不爆仓或负收益迭代的？</b></summary>

Reef 架构设计了 **5 重硬性交付门禁 (5 Commit Gates)** 与 **自动备份回滚机制**：
1. **严格因果隔离 (No-Lookahead)**：任何变异进化公式必须通过逐 bar 递增因果检验，杜绝时序未来函数泄露；
2. **方差健康度检验 (Non-Degenerate)**：因子截面标准差 $\sigma \ge 1\times 10^{-4}$，防止因子在震荡市输出常数或退化为 Beta；
3. **样本外严苛超额 (Score Improvement)**：仅当候选策略在历史高信度实盘轨迹与验证集上的综合得分超越基准策略至少 $+3\%$ 时才允许晋级；
4. **最大回撤硬约束 (Drawdown Guard)**：新策略的最大回撤不得劣于基准策略的 1.05 倍，杜绝激进放大的下行风险；
5. **栈机沙箱全量重放 (StackVM Sandbox)**：在虚拟机沙箱中全量重放，确认零除零、零 NaN、零越界；
6. **归档备份与秒级回滚 (Archive Backup)**：任何热替换前均会将当前运行的策略自动备份至 `strategies/archive/`，支持秒级无损回滚。
</details>

---

## 🙏 致谢 / Acknowledgements

特别感谢 [Human-Agent-Society/reef](https://github.com/Human-Agent-Society/reef) 团队在持续自进化 Agent 基础设施（Continual learning infra for self-improving agents）领域的卓越开源工作，为本项目的持续演化与策略自迭代机制提供了极具价值的灵感与架构参考！

---

## 📄 License

本项目仅供学习和研究用途。使用者需遵守所在地区的法律法规，自行承担使用风险。

---

**与 OKX 官方无隶属或背标关系 · 不构成投资建议 · 加密货币交易高风险**
