# 七星高照ETF轮动 — 本地化与优化工作总结

> 项目：聚宽云端 ETF 轮动策略（V1.7 母版）的本地化移植与优化
> 阶段状态（2026-07-03 修订）：
> - 阶段一：V1.7 本地化对齐 —— **通过**
> - 阶段二：参数初筛 —— **完成**
> - 阶段三：C+E 结构性假设 —— **待严格验证**（前期结论已撤回，见下方说明）
> - 阶段四：实盘化 —— **尚未开始正式验收**

---

## 一、项目背景

将聚宽云端的「七星高照ETF轮动策略 V1.7 母版」移植到本地回测引擎运行，经过对齐验证后进入优化阶段，探索参数调优与结构性改进。

## 二、本地化对齐阶段（通过）

### 核心成果
- **性能优化**：43s/日 → 7s/日（6.6x 提升）
- **母版对齐**：全年 242 天回测，选股 0 分歧，价格 100% 在 0.01 元以内（61% 完全一致）
- **剩余差异**：98 笔全部是分钟 K 线撮合精度的蝴蝶效应，已达回测引擎理论上限
- **本地最终权益**：1,345,491.04（与聚宽母版一致）

### 关键修复（固化在 run_local.py）

| 修复项 | 说明 |
|--------|------|
| 活跃证券集合注册 | post_exec_hook 限制快照查询范围至 ETF 池 |
| attribute_history 多字段 bug | 拆分单字段查询再合并，绕过引擎数据检索 bug |
| attribute_history 强制 fq='pre' | 聚宽 use_real_price=True 对应动态前复权 |
| load_1d_etf 缓存污染修复 | monkey-patch 强制全列读取 + 清空 _ETF_YEAR_CACHE |
| 511880/501018 佣金率 0.0002 | monkey-patch 修复佣金率 |
| get_name 内存缓存 | 走 get_security_info 避免全市场快照 |
| 501018 LOF 分类 + price_decimals | 50 开头 LOF 识别为 etf + 3 位小数报价 |

### 工程约定
- 所有引擎修改隔离在 run_local.py，不修改引擎源码
- 优化必须保持与其他策略的兼容性（不硬编码证券列表）

### 已知局限
- 当前对齐在 `parity` 模式下完成（临时关闭 ETF 滑点、放宽资金约束、用特定分钟 close 成交），不能直接推导为实盘表现
- 14:00 分钟 bar 的时间语义（标签代表区间开始还是结束）尚未正式审计
- 剩余 98 笔差异的原始诊断 JSON 未入仓库，外部人员无法独立复核

## 三、参数初筛阶段（完成）

### 阶段1：参数敏感性扫描（方向A+B）

对 5 个关键参数进行单变量全年回测扫描，并用 Walk-forward（IS 2022-2023 → OOS 2024）验证。

**结论：在单变量扫描范围内，baseline 参数在原池中表现较优。**

| 参数 | 判定 | 证据 |
|------|------|------|
| lookback_days=25 | 尖峰最优 | 邻近值断崖式下降（20→8.82%, 30→5.43%） |
| holdings_num=1 | 单调最优 | 分散持仓单调降低收益 |
| short_lookback_days=10 | baseline 最优 | — |
| profit_protection_threshold=0.05 | baseline 最优 | — |
| profit_protection_lookback=1 | Walk-forward 验证最优 | 看似可改进实则过拟合 |

**重要限定**：单变量扫描不能证明 baseline 是全局最优，参数间存在交互（大池中 25 日可能不是最优，Top2 可能降低 QDII 风险）。多参数联合扫描尚未进行。

**关键教训**：profit_protection_lookback=15 在 2024 单年收益 46.02%（看似巨大改进），但 2022-2023 仅 52.26%（baseline 64.01%），IS/OOS 排序反转，典型的过拟合信号。

### 阶段2：结构性改进（方向C+E）—— 结论已撤回

> ⚠️ **重要警告**：方向 C（大池）和 C+E 组合的实验结论已被撤回。
> 原因：NAV 数据缺失处理存在严重 bug（fail-open），导致大池中 29/37 只无净值数据的 ETF 绕过溢价过滤。
> 详见下方"P0 修复说明"。

#### 撤回的结论
以下结论此前被声明为"已验证有效"，现因实验污染撤回：
- ~~C+E 组合 OOS 收益 91.24%，非过拟合~~
- ~~C+E 应成为新 baseline~~
- ~~baseline 参数是全局最优~~

#### 保留的结论
- E 方向（multi_period）在原池中是干净的（0 笔 NAV 污染），值得继续验证
- vol_adjusted 方向失败（波动率惩罚过严，收益暴跌）

## 四、P0 修复说明（2026-07-03）

经严格评审，发现前期优化实验存在以下严重问题，已修复：

### P0-1~3: NAV 缺失处理污染（最严重）

**问题**：run_local.py 第 382-390 行将 NAV 缺失的 ETF 溢价率兜底为 0（fail-open），导致无净值数据的 ETF 自动通过溢价过滤。

**影响范围**：
- 大池 37 只 ETF（POOL_BAK，去掉缺失的 159201）中仅 9 只有 NAV 数据，**28 只缺失**（75.7%）
- 防御 ETF 511880 有 NAV 数据（单独列示，不在风险池中）
- NAV 文件时间范围仅 2023-10-09 ~ 2024-12-31，2022 全年及 2023 Q1-Q3 全部缺失
- 大池+baseline IS：150 笔买入中 **84 笔（56%）买入无 NAV 的 ETF**（invalid）
- 大池+multi_period (C+E) IS：147 笔买入中 **80 笔（54%）买入无 NAV 的 ETF**（invalid）
- 原池+multi_period (E) OOS 2024：53 笔买入中 **0 笔污染**（valid，可保留为候选）
- 原池+multi_period (E) IS 2022-2023：因 NAV 覆盖不足，需标记 needs_rerun

**结论**：大池实验的收益提升（74.72%/96.89%）混杂了"绕过溢价过滤"效应，不能作为可靠优化证据。详见 qixing_optimize/audits/INVALIDATED_RESULTS.md。

### P0-4: Calmar 计算错误

**问题**：run_sweep.py 使用 `total_return / max_dd` 而非标准 `年化收益 / max_dd`。

**影响**：对 2022-2023 两年期 IS 区间，Calmar 被高估约 2 倍。

**修复**：改为 `(1+total_return)^(252/n_days) - 1` / max_dd，新增 `annualized_return_pct` 和 `n_trade_days` 字段。

### P0-5: NAV 缺失改为三种模式 + 回测前预检

**修复**：
1. run_local.py 新增 `nav_mode` 参数（默认 `parity`），支持三种 NAV 缺失处理策略：
   - `parity`：NAV 缺失立即终止回测（数据不完整时无法宣称对齐聚宽）
   - `realistic`：NAV 缺失返回 None，策略原生逻辑跳过该 ETF（研究/实盘模式）
   - `legacy_invalid`：保留旧版 0 溢价兜底（仅用于复现旧污染结果，结果标记 `invalid_due_to_nav_fail_open=true`）
2. 新增 NAV 回测前预检（`_run_nav_precheck`）：在 post_exec_hook 末尾、daily loop 之前检查池中所有 ETF 的 NAV 覆盖，生成 `qixing_optimize/audits/NAV_COVERAGE_REPORT.md` 和 `nav_coverage.json`
3. 运行时缺失统计（`_build_nav_report`）：记录实际调用 `get_premium_rate` 时的缺失事件

## 五、文件结构

```
七星高照ETF轮动/
├── 七星高照ETF轮动策略v1.7-母版.py      # 策略源码（聚宽母版，未修改）
├── run_local.py                          # 本地运行器（所有 patch + 优化注入 + NAV 三模式）
├── SUMMARY.md                            # 项目总结（P0 修订版）
├── tests/
│   └── test_calmar.py                    # Calmar 计算单元测试
├── qixing_optimize/
│   ├── scripts/
│   │   ├── run_sweep.py                  # 参数/策略扫描脚本（Calmar 已修复）
│   │   └── analyze_results.py            # 结果分析脚本
│   ├── audits/
│   │   ├── NAV_COVERAGE_REPORT.md        # NAV 覆盖率预检报告
│   │   ├── nav_coverage.json             # NAV 覆盖率结构化数据
│   │   ├── INVALIDATED_RESULTS.md        # 旧实验污染审计报告
│   │   └── invalidated_results.json      # 旧实验污染结构化数据
│   └── runs/
│       ├── WALK_FORWARD_REPORT.md        # 参数扫描报告（方向A+B，结论已加限定）
│       └── DIRECTION_CE_REPORT.md        # 结构性改进报告（方向C+E，结论已撤回）
```

## 六、经验总结

1. **baseline 参数在单变量扫描中较优**，但不能证明是全局最优（参数交互未覆盖）
2. **短区间回测不能判断参数优劣**：1月功能验证的结论与全年回测完全相反
3. **Walk-forward 验证是识别过拟合的关键**：单年回测的"改进"可能是对特定年份市场特征的适配
4. **类型安全很重要**：int 参数传 float 会导致引擎行为异常且不易察觉
5. **fail-closed 优于 fail-open**：数据缺失时应排除该标的，而非放行（P0 教训）
6. **单年 holdout 不等于滚动 walk-forward**：2024 已用于发现参数，不能再称独立 OOS

## 七、后续方向（P0 完成后）

### P0 剩余项（待完成）
- P0-6: 验证 14:00 分钟 bar 时间语义
- P0-7: 拆分 parity 与 realistic 两种成交模式
- P0-8: 添加 requirements.txt + 环境配置说明

### P1: 重做扩池实验（四组严格消融）
| 组别 | 池 | 得分 |
|------|-----|------|
| A | 原7只 | 25日 |
| B | 原7只 | 25+60日 |
| C | 时间点一致大池 | 25日 |
| D | 时间点一致大池 | 25+60日 |

所有组必须使用同样的 NAV 处理（fail-closed）、成本、可交易性规则。

### P2: 真正滚动 walk-forward
至少 5+ 滚动窗口（2018-2020→2021 起每滚动一年）。

### P3: 实盘压力测试
滑点（5/10/20 bp）、延迟（1/5 分钟）、不同时点、QDII 无法成交场景、成交额占比限制等。

**在 P0 剩余项、P1、P2 完成前，不要将 C+E 设为实盘 baseline，也不要在 C+E 上继续参数扫描。**
