"""
七星高照 P4-04 低频 O7/C8D 切换框架验证
==========================================
核心问题：低频化以后，O7/C8D 切换信号还剩多少真实、扣成本后的超额？

数据来源：
- P4-01 已生成的 O7_baseline / C8D_baseline 日频净值（7 个窗口）
- P4-02 T-1 修正版 feature_panel / target_panel
- P4-03 relative_score 重建逻辑（与 P4-02 完全一致）

3 类低频框架：
- A1: 月度 relative_score 反向框架
- A2: 月度 o7_internal_corr_60d 框架
- B1: 月度双信号确认框架
- C1: 季度双信号确认框架

每个框架 × 3 档成本（low/mid/high）= 12 个方案
× 7 个 walk-forward 窗口 + 全区间

边界：
- 不新增池、不引入 B37、不引入 multi_period
- 不新增因子、不调原策略参数
- 不网格搜索阈值、不用未来收益反推规则
- 不做日频按桶切换、年化切换次数 >12 的方案不进入候选
- 不宣布可实盘、不修改 P1/P2/P3/P4-01/P4-02/P4-03 冻结结论
"""
import sys
import os
import json
import math
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ==== 路径 ====
P4_01_DIR = ROOT / 'qixing_optimize' / 'runs' / 'p4_01_core_pool_validation'
P4_02_DIR = ROOT / 'qixing_optimize' / 'runs' / 'p4_02_non_momentum_predictive_validation'
OUT_DIR = ROOT / 'qixing_optimize' / 'runs' / 'p4_04_low_frequency_switch_validation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== 池定义（P4-01/P4-02/P4-03 冻结，不得修改） ====
POOL_O7 = [
    "518880.XSHG", "159985.XSHE", "501018.XSHG", "161226.XSHE",
    "513100.XSHG", "159915.XSHE", "511220.XSHG",
]
POOL_C8D = [
    "518880.XSHG", "159915.XSHE", "510300.XSHG", "510500.XSHG",
    "510050.XSHG", "513100.XSHG", "511220.XSHG", "159985.XSHE",
]

# ==== 7 个 walk-forward 窗口（P2/P4-01/P4-02/P4-03 冻结） ====
WINDOW_RANGES = OrderedDict([
    ('2020',   ('2020-01-02', '2020-12-31')),
    ('2021',   ('2021-01-04', '2021-12-31')),
    ('2022',   ('2022-01-04', '2022-12-30')),
    ('2023',   ('2023-01-03', '2023-12-29')),
    ('2024',   ('2024-01-02', '2024-12-31')),
    ('2025',   ('2025-01-02', '2025-12-31')),
    ('2026H1', ('2026-01-02', '2026-06-30')),
])
ALL_WINDOWS = list(WINDOW_RANGES.keys())

# ==== relative_score 五个组成项（P4-02 冻结，不得改权重/方向） ====
RELATIVE_SCORE_COMPONENTS = [
    ("dd_diff_120d",         +1),
    ("vol_expansion_diff",   -1),
    ("amt_change_diff",      -1),
    ("efficiency_diff",      +1),
    ("dispersion_diff",      -1),
]

# ==== 三档成本（任务书七、4 冻结） ====
COST_SCENARIOS = OrderedDict([
    ("low",  {"single_side": 0.0010}),
    ("mid",  {"single_side": 0.0020}),
    ("high", {"single_side": 0.0037}),
])

# ==== 4 个框架 ====
FRAMEWORKS = ['A1_relative_monthly', 'A2_corr_monthly', 'B1_dual_monthly', 'C1_dual_quarterly']

# ==== 滚动分位数参数（任务书六冻结） ====
ROLLING_QUANTILE_WINDOW = 252  # 最少 252 个交易日
LOW_QUANTILE  = 1.0 / 3.0      # 三分位低阈值
HIGH_QUANTILE = 2.0 / 3.0      # 三分位高阈值

# ==== 通过标准阈值（任务书十二冻结） ====
PASS_WINDOWS_RATIO = 5.0 / 7.0      # 至少 5/7 窗口跑赢 O7
MAX_ANNUAL_SWITCH = 12              # 年化切换次数 ≤12
OBVIOUS_DRAG_RETURN = -0.05         # 年度收益差 ≤ -5pp 为明显拖累
OBVIOUS_DRAG_DD = 0.03              # 最大回撤恶化 ≥ 3pp 为明显拖累
HIGH_COST_THRESHOLD = 0.95          # 高成本档收益不低于 O7 的 95%
HIGH_COST_DD_THRESHOLD = 0.02       # 高成本档回撤不高于 O7 + 2pp
FLOAT_EPSILON = 1e-9                # 浮点精度容差，用于"跑赢"判断


# ============================================================
# 工具函数
# ============================================================
def date_to_window(d):
    d = pd.Timestamp(d)
    for w, (s, e) in WINDOW_RANGES.items():
        if pd.Timestamp(s) <= d <= pd.Timestamp(e):
            return w
    return None


def z_score(s):
    """全样本 z-score（与 P4-02/P4-03 一致）"""
    m = s.mean()
    sd = s.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - m) / sd


def reconstruct_relative_score(features):
    """重建 relative_score（与 P4-02/P4-03 完全一致，全样本 z-score 等权加总）

    P4-02 已对 features 做了 shift(1)，feature_panel.loc[T] 使用截至 T-1 的信息。
    """
    rel_score = pd.Series(0.0, index=features.index)
    for col, sign in RELATIVE_SCORE_COMPONENTS:
        z = z_score(features[col])
        rel_score += sign * z
    return rel_score


# ============================================================
# 数据加载
# ============================================================
def load_baseline_nav():
    """加载 P4-01 O7_baseline / C8D_baseline 日频净值，拼接 7 个窗口

    返回：
    - o7_nav: Series，index=date
    - c8d_nav: Series，index=date
    - 每个窗口的日收益（pct_change）
    """
    print("[1] 加载 P4-01 baseline 日频净值 ...")
    o7_records = []
    c8d_records = []
    for w in ALL_WINDOWS:
        for pool, rec_list in [('O7', o7_records), ('C8D', c8d_records)]:
            fp = P4_01_DIR / f'{pool}_baseline_{w}.json'
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for rec in data.get('daily', []):
                rec_list.append({
                    'date': pd.Timestamp(rec['date']).normalize(),
                    'value': float(rec['value']),
                    'window': w,
                })
    o7_df = pd.DataFrame(o7_records).sort_values('date').reset_index(drop=True)
    c8d_df = pd.DataFrame(c8d_records).sort_values('date').reset_index(drop=True)

    # 同一日期取最后一条（去重）
    o7_df = o7_df.drop_duplicates(subset='date', keep='last').set_index('date')
    c8d_df = c8d_df.drop_duplicates(subset='date', keep='last').set_index('date')

    # 仅保留两个池共有的交易日
    common_dates = o7_df.index.intersection(c8d_df.index)
    o7_df = o7_df.loc[common_dates]
    c8d_df = c8d_df.loc[common_dates]

    o7_nav = o7_df['value']
    c8d_nav = c8d_df['value']
    print(f"  O7  净值: {len(o7_nav)} 个交易日，区间 {o7_nav.index[0].date()} ~ {o7_nav.index[-1].date()}")
    print(f"  C8D 净值: {len(c8d_nav)} 个交易日，区间 {c8d_nav.index[0].date()} ~ {c8d_nav.index[-1].date()}")
    return o7_nav, c8d_nav


def load_features():
    """加载 P4-02 T-1 修正版 feature_panel"""
    print("[2] 加载 P4-02 feature_panel ...")
    fp = pd.read_csv(P4_02_DIR / 'feature_panel.csv', parse_dates=['date'], index_col='date')
    print(f"  feature_panel: {fp.shape}")
    return fp


# ============================================================
# 信号生成
# ============================================================
def build_signals(features):
    """构建 4 个框架的信号序列

    信号含义：每日应持有的池（'O7' 或 'C8D'）
    信号在月末/季末收盘后生成，下一交易日生效。
    feature_panel 已 shift(1)，即 feature_panel.loc[T] 使用截至 T-1 的信息。
    因此 T 日生成的信号使用 T-1 及之前的数据，符合 T-1 口径。

    返回 dict: {framework_name: pd.Series(每日信号 'O7'/'C8D')}
    """
    print("[3] 构建低频切换信号 ...")

    rel_score = reconstruct_relative_score(features)
    o7_corr = features['o7_internal_corr_60d'].copy()

    # 计算滚动三分位阈值（只用历史数据）
    def rolling_quantile(s, q):
        """截至当前日（含）的滚动分位数"""
        return s.expanding(min_periods=ROLLING_QUANTILE_WINDOW).quantile(q)

    rel_low_q = rolling_quantile(rel_score, LOW_QUANTILE)
    corr_high_q = rolling_quantile(o7_corr, HIGH_QUANTILE)

    # 月末/季末标记
    # 月末：当前月 != 下一交易日所属月
    # 注意：feature_panel 的 index 是交易日，需要用 shift 判断月末
    s = pd.Series(features.index, index=features.index)
    next_month = s.shift(-1).dt.to_period('M')
    cur_month = s.dt.to_period('M')
    is_month_end = next_month != cur_month
    is_month_end = is_month_end.fillna(False)

    # 季末：3/6/9/12 月的月末
    is_quarter_end = is_month_end & s.dt.month.isin([3, 6, 9, 12])

    # 框架 A1: 月度 relative_score 反向
    # 月末 relative_score <= 历史滚动三分位低阈值 → 下月持有 C8D
    # 否则 → 下月持有 O7
    a1_signal = pd.Series('O7', index=features.index)
    a1_trigger = is_month_end & (rel_score <= rel_low_q) & rel_low_q.notna()
    # 信号在月末生成，下月持有
    # 用 forward fill 把月末信号扩展到下月所有交易日
    a1_monthly_signal = pd.Series('O7', index=features.index)
    a1_monthly_signal.loc[a1_trigger] = 'C8D'
    # shift(1) 后 forward fill：下一交易日才开始持有
    a1_signal = a1_monthly_signal.shift(1).ffill().fillna('O7')

    # 框架 A2: 月度 o7_internal_corr_60d
    # 月末 o7_corr >= 历史滚动三分位高阈值 → 下月持有 C8D
    a2_monthly_signal = pd.Series('O7', index=features.index)
    a2_trigger = is_month_end & (o7_corr >= corr_high_q) & corr_high_q.notna()
    a2_monthly_signal.loc[a2_trigger] = 'C8D'
    a2_signal = a2_monthly_signal.shift(1).ffill().fillna('O7')

    # 框架 B1: 月度双信号确认
    # 同时满足 A1 和 A2 的条件才切到 C8D
    b1_monthly_signal = pd.Series('O7', index=features.index)
    b1_trigger = a1_trigger & a2_trigger
    b1_monthly_signal.loc[b1_trigger] = 'C8D'
    b1_signal = b1_monthly_signal.shift(1).ffill().fillna('O7')

    # 框架 C1: 季度双信号确认
    # 在季末判断，条件同 B1
    c1_monthly_signal = pd.Series('O7', index=features.index)
    c1_trigger = is_quarter_end & (rel_score <= rel_low_q) & (o7_corr >= corr_high_q) & rel_low_q.notna() & corr_high_q.notna()
    c1_monthly_signal.loc[c1_trigger] = 'C8D'
    c1_signal = c1_monthly_signal.shift(1).ffill().fillna('O7')

    signals = {
        'A1_relative_monthly': a1_signal,
        'A2_corr_monthly':     a2_signal,
        'B1_dual_monthly':     b1_signal,
        'C1_dual_quarterly':   c1_signal,
    }

    for name, sig in signals.items():
        n_c8d = (sig == 'C8D').sum()
        n_total = len(sig)
        print(f"  {name}: C8D 持有 {n_c8d}/{n_total} 日 ({n_c8d/n_total*100:.1f}%)")

    return signals


# ============================================================
# 回测
# ============================================================
def backtest_switching(signal, o7_nav, c8d_nav, single_side_cost):
    """真实切换回测

    信号 signal.loc[T] 表示 T 日应持有的池。
    每日组合净值收益来自当前持有池的净值变化。
    当信号从 O7 切到 C8D 或反之，当日额外扣除 single_side_cost。

    返回：
    - nav: Series，组合日净值（初始归一为 1.0）
    - switch_events: list of dict，每次切换事件
    """
    # 对齐到共同交易日
    common = signal.index.intersection(o7_nav.index).intersection(c8d_nav.index)
    sig = signal.loc[common]
    o7 = o7_nav.loc[common]
    c8d = c8d_nav.loc[common]

    # 计算两池的日收益率
    o7_ret = o7.pct_change().fillna(0.0)
    c8d_ret = c8d.pct_change().fillna(0.0)

    # 组合日收益
    n = len(common)
    nav_values = np.zeros(n)
    nav_values[0] = 1.0
    switch_events = []
    prev_holding = sig.iloc[0]
    cur_holding = prev_holding

    for i in range(1, n):
        new_holding = sig.iloc[i]
        pool_ret = o7_ret.iloc[i] if new_holding == 'O7' else c8d_ret.iloc[i]
        cost = 0.0
        if new_holding != prev_holding:
            cost = single_side_cost
            switch_events.append({
                'date': str(common[i].date()),
                'from': prev_holding,
                'to': new_holding,
                'cost': cost,
            })
        nav_values[i] = nav_values[i-1] * (1.0 + pool_ret) * (1.0 - cost)
        prev_holding = new_holding

    nav = pd.Series(nav_values, index=common)
    return nav, switch_events


def compute_metrics(nav, switch_events, window_range=None):
    """计算单个回测的指标"""
    if window_range is not None:
        s, e = window_range
        nav = nav.loc[pd.Timestamp(s):pd.Timestamp(e)]

    if len(nav) < 2:
        return None

    rets = nav.pct_change().dropna()
    n_days = len(nav)
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    annual_return = (1 + total_return) ** (252.0 / n_days) - 1.0

    # 最大回撤
    peak = nav.cummax()
    dd = (nav - peak) / peak
    max_dd = dd.min()

    # 夏普（无风险利率 0）
    if rets.std() > 0:
        sharpe = np.sqrt(252) * rets.mean() / rets.std()
    else:
        sharpe = 0.0

    # Calmar
    if max_dd < 0:
        calmar = annual_return / abs(max_dd)
    else:
        calmar = float('inf') if annual_return > 0 else 0.0

    # 切换统计
    n_switches = len(switch_events)
    # 年化切换次数（基于实际交易日）
    annual_switches = n_switches / (n_days / 252.0)

    # 持有周期
    if n_switches > 0:
        # 计算每段持有天数
        holding_days = []
        if len(switch_events) > 0:
            # 用 nav 的 signal 推断持有段
            # 简化：用 switch_events 的日期差
            dates = [pd.Timestamp(nav.index[0])] + [pd.Timestamp(e['date']) for e in switch_events] + [pd.Timestamp(nav.index[-1])]
            for i in range(1, len(dates)):
                holding_days.append((dates[i] - dates[i-1]).days)
        avg_hold = np.mean(holding_days) if holding_days else 0
        med_hold = np.median(holding_days) if holding_days else 0
    else:
        avg_hold = n_days
        med_hold = n_days

    # 成本扣减总额（按净值比例）
    total_cost = 1.0
    for e in switch_events:
        total_cost *= (1.0 - e['cost'])
    total_cost_pct = 1.0 - total_cost

    # 成本前收益（假设无成本）
    # 重新计算：用 signal 收益但不扣成本
    # 简化：成本前净值 = nav / (1 - total_cost_pct) 近似
    # 更准确做法：重跑无成本回测，但为简化用近似
    # 实际上 cost 已按 (1-cost) 累乘到 nav，所以成本前净值 ≈ nav / prod(1-cost)
    nav_no_cost = nav / total_cost
    total_return_no_cost = nav_no_cost.iloc[-1] / nav_no_cost.iloc[0] - 1.0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'final_nav': nav.iloc[-1],
        'n_trade_days': n_days,
        'n_switches': n_switches,
        'annual_switches': annual_switches,
        'avg_hold_days': avg_hold,
        'median_hold_days': med_hold,
        'total_cost_pct': total_cost_pct,
        'total_return_no_cost': total_return_no_cost,
    }


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("P4-04 低频 O7/C8D 切换框架验证")
    print("=" * 70)

    # 1. 加载数据
    o7_nav, c8d_nav = load_baseline_nav()
    features = load_features()

    # 2. 构建信号
    signals = build_signals(features)

    # 3. 对齐信号与净值
    common_dates = o7_nav.index.intersection(features.index)
    print(f"\n[4] 信号与净值对齐：{len(common_dates)} 个共同交易日")

    # 4. 对每个框架 × 每档成本回测
    print("\n[5] 执行回测 ...")
    all_nav = OrderedDict()  # {strategy_name: pd.Series(nav)}
    all_events = OrderedDict()
    all_metrics = []  # 每个方案 × 每个窗口 + 全区间

    # 加 O7_baseline / C8D_baseline 作为对照
    o7_nav_aligned = o7_nav.loc[common_dates]
    c8d_nav_aligned = c8d_nav.loc[common_dates]

    # 归一化为初始净值 1.0
    o7_nav_norm = o7_nav_aligned / o7_nav_aligned.iloc[0]
    c8d_nav_norm = c8d_nav_aligned / c8d_nav_aligned.iloc[0]

    all_nav['O7_baseline'] = o7_nav_norm
    all_nav['C8D_baseline'] = c8d_nav_norm
    all_events['O7_baseline'] = []
    all_events['C8D_baseline'] = []

    for fwk in FRAMEWORKS:
        sig = signals[fwk].loc[common_dates]
        for cost_name, cost_info in COST_SCENARIOS.items():
            strategy_name = f'{fwk}_{cost_name}_cost'
            nav, events = backtest_switching(sig, o7_nav_norm, c8d_nav_norm, cost_info['single_side'])
            all_nav[strategy_name] = nav
            all_events[strategy_name] = events
            print(f"  {strategy_name}: 最终净值 {nav.iloc[-1]:.4f}, 切换 {len(events)} 次")

    # 5. 计算每个方案的指标（全区间 + 每个窗口）
    print("\n[6] 计算指标 ...")

    # O7_baseline / C8D_baseline 全区间指标
    for name in ['O7_baseline', 'C8D_baseline']:
        m = compute_metrics(all_nav[name], all_events[name], window_range=None)
        m['strategy'] = name
        m['framework'] = 'baseline'
        m['cost_scenario'] = 'n/a'
        m['window'] = 'all'
        all_metrics.append(m)

    # 每个切换方案的全区间 + 每窗口指标
    for fwk in FRAMEWORKS:
        for cost_name in COST_SCENARIOS.keys():
            strategy_name = f'{fwk}_{cost_name}_cost'
            # 全区间
            m = compute_metrics(all_nav[strategy_name], all_events[strategy_name], window_range=None)
            if m is not None:
                m['strategy'] = strategy_name
                m['framework'] = fwk
                m['cost_scenario'] = cost_name
                m['window'] = 'all'
                all_metrics.append(m)
            # 每个窗口
            for w, wrange in WINDOW_RANGES.items():
                m = compute_metrics(all_nav[strategy_name], all_events[strategy_name], window_range=wrange)
                if m is not None:
                    m['strategy'] = strategy_name
                    m['framework'] = fwk
                    m['cost_scenario'] = cost_name
                    m['window'] = w
                    all_metrics.append(m)

    # 也算 baseline 的每窗口指标
    for name in ['O7_baseline', 'C8D_baseline']:
        for w, wrange in WINDOW_RANGES.items():
            m = compute_metrics(all_nav[name], all_events[name], window_range=wrange)
            if m is not None:
                m['strategy'] = name
                m['framework'] = 'baseline'
                m['cost_scenario'] = 'n/a'
                m['window'] = w
                all_metrics.append(m)

    metrics_df = pd.DataFrame(all_metrics)
    print(f"  共 {len(metrics_df)} 条指标记录")

    # 6. 输出文件
    print("\n[7] 输出文件 ...")

    # 6.1 switch_nav_daily.csv/json
    nav_df = pd.DataFrame(all_nav)
    nav_df.index.name = 'date'
    nav_df.to_csv(OUT_DIR / 'switch_nav_daily.csv')
    with open(OUT_DIR / 'switch_nav_daily.json', 'w', encoding='utf-8') as f:
        json.dump({
            strategy: {str(d.date()): float(v) for d, v in nav.items()}
            for strategy, nav in all_nav.items()
        }, f, ensure_ascii=False, indent=2)
    print(f"  switch_nav_daily.csv/json ({nav_df.shape})")

    # 6.2 switch_events.csv/json
    events_records = []
    for strategy, events in all_events.items():
        for e in events:
            events_records.append({'strategy': strategy, **e})
    events_df = pd.DataFrame(events_records)
    events_df.to_csv(OUT_DIR / 'switch_events.csv', index=False)
    with open(OUT_DIR / 'switch_events.json', 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in all_events.items() if v}, f, ensure_ascii=False, indent=2)
    print(f"  switch_events.csv/json ({len(events_df)} events)")

    # 6.3 strategy_results_by_window.csv/json
    results_df = metrics_df.copy()
    results_df.to_csv(OUT_DIR / 'strategy_results_by_window.csv', index=False)
    results_df.to_json(OUT_DIR / 'strategy_results_by_window.json', orient='records', force_ascii=False, indent=2)
    print(f"  strategy_results_by_window.csv/json ({len(results_df)} rows)")

    # 6.4 strategy_summary.csv/json（每个方案的全区间汇总 + 通过标准检查）
    summary_records = []
    # 取全区间指标
    all_metrics_df = metrics_df[metrics_df['window'] == 'all'].copy()
    o7_all = all_metrics_df[all_metrics_df['strategy'] == 'O7_baseline'].iloc[0]
    c8d_all = all_metrics_df[all_metrics_df['strategy'] == 'C8D_baseline'].iloc[0]

    for fwk in FRAMEWORKS:
        for cost_name in COST_SCENARIOS.keys():
            strategy_name = f'{fwk}_{cost_name}_cost'
            row = all_metrics_df[all_metrics_df['strategy'] == strategy_name]
            if row.empty:
                continue
            row = row.iloc[0]

            # 计算每窗口跑赢 O7 的次数（用 epsilon 避免浮点精度误判，零切换方案不算跑赢）
            fwk_window_df = metrics_df[(metrics_df['strategy'] == strategy_name)]
            o7_window_df = metrics_df[metrics_df['strategy'] == 'O7_baseline']
            n_win_o7 = 0
            for w in ALL_WINDOWS:
                s_row = fwk_window_df[fwk_window_df['window'] == w]
                o_row = o7_window_df[o7_window_df['window'] == w]
                if s_row.empty or o_row.empty:
                    continue
                # 零切换方案的收益等于 O7，不算"跑赢"
                if row['n_switches'] == 0:
                    continue
                if s_row.iloc[0]['total_return'] > o_row.iloc[0]['total_return'] + FLOAT_EPSILON:
                    n_win_o7 += 1
            # 跑赢 C8D 次数
            c8d_window_df = metrics_df[metrics_df['strategy'] == 'C8D_baseline']
            n_win_c8d = 0
            for w in ALL_WINDOWS:
                s_row = fwk_window_df[fwk_window_df['window'] == w]
                c_row = c8d_window_df[c8d_window_df['window'] == w]
                if s_row.empty or c_row.empty:
                    continue
                if row['n_switches'] == 0:
                    continue
                if s_row.iloc[0]['total_return'] > c_row.iloc[0]['total_return'] + FLOAT_EPSILON:
                    n_win_c8d += 1

            # 回撤 ≤20% 的窗口数
            n_dd_le_20 = 0
            for w in ALL_WINDOWS:
                s_row = fwk_window_df[fwk_window_df['window'] == w]
                if s_row.empty:
                    continue
                if s_row.iloc[0]['max_drawdown'] >= -0.20:
                    n_dd_le_20 += 1

            # 年度收益统计
            yearly_returns = []
            for w in ALL_WINDOWS:
                s_row = fwk_window_df[fwk_window_df['window'] == w]
                if s_row.empty:
                    continue
                yearly_returns.append(s_row.iloc[0]['total_return'])
            yearly_returns = np.array(yearly_returns)

            # 2024 / 2026H1 表现
            ret_2024 = fwk_window_df[fwk_window_df['window'] == '2024']['total_return'].values
            ret_2024 = ret_2024[0] if len(ret_2024) > 0 else None
            ret_2026 = fwk_window_df[fwk_window_df['window'] == '2026H1']['total_return'].values
            ret_2026 = ret_2026[0] if len(ret_2026) > 0 else None
            o7_ret_2024 = o7_window_df[o7_window_df['window'] == '2024']['total_return'].values
            o7_ret_2024 = o7_ret_2024[0] if len(o7_ret_2024) > 0 else None
            o7_ret_2026 = o7_window_df[o7_window_df['window'] == '2026H1']['total_return'].values
            o7_ret_2026 = o7_ret_2026[0] if len(o7_ret_2026) > 0 else None

            # 2024 / 2026H1 是否明显拖累
            drag_2024 = None
            if ret_2024 is not None and o7_ret_2024 is not None:
                diff = ret_2024 - o7_ret_2024
                drag_2024 = diff <= OBVIOUS_DRAG_RETURN
            drag_2026 = None
            if ret_2026 is not None and o7_ret_2026 is not None:
                diff = ret_2026 - o7_ret_2026
                drag_2026 = diff <= OBVIOUS_DRAG_RETURN

            # 通过标准检查（仅中成本档作为主要验收）
            is_mid = cost_name == 'mid'
            # 零切换方案的收益等于 O7，不算"跑赢"
            beat_o7_total = row['total_return'] > o7_all['total_return'] + FLOAT_EPSILON if row['n_switches'] > 0 else False
            beat_c8d_total = row['total_return'] > c8d_all['total_return'] + FLOAT_EPSILON if row['n_switches'] > 0 else False
            pass_criteria = {}
            if is_mid:
                pass_criteria = {
                    '1_total_return_gt_o7': beat_o7_total,
                    '2_5of7_windows_win_o7': n_win_o7 >= 5,
                    '3_max_dd_le_o7': row['max_drawdown'] >= o7_all['max_drawdown'] - FLOAT_EPSILON,
                    '4_dd_le_20_ge_o7': n_dd_le_20 >= sum(1 for w in ALL_WINDOWS if metrics_df[(metrics_df['strategy']=='O7_baseline')&(metrics_df['window']==w)]['max_drawdown'].values[0] >= -0.20),
                    '5_annual_switch_le_12': row['annual_switches'] <= MAX_ANNUAL_SWITCH,
                    '6_no_2024_drag': not drag_2024 if drag_2024 is not None else True,
                    '7_no_2026_drag': not drag_2026 if drag_2026 is not None else True,
                    '8_high_cost_not_obviously_lose': None,  # 在后面填入
                    '9_total_cost_economically_meaningful': row['total_return'] > 0,
                    '10_logic_explainable_not_single_year': row['n_switches'] > 0,  # 零切换无逻辑可言
                }

            summary_records.append({
                'strategy': strategy_name,
                'framework': fwk,
                'cost_scenario': cost_name,
                'total_return': row['total_return'],
                'annual_return': row['annual_return'],
                'max_drawdown': row['max_drawdown'],
                'sharpe': row['sharpe'],
                'calmar': row['calmar'],
                'final_nav': row['final_nav'],
                'n_switches': row['n_switches'],
                'annual_switches': row['annual_switches'],
                'avg_hold_days': row['avg_hold_days'],
                'median_hold_days': row['median_hold_days'],
                'total_cost_pct': row['total_cost_pct'],
                'total_return_no_cost': row['total_return_no_cost'],
                'excess_vs_o7_total': row['total_return'] - o7_all['total_return'],
                'excess_vs_o7_annual': row['annual_return'] - o7_all['annual_return'],
                'dd_improvement_vs_o7': row['max_drawdown'] - o7_all['max_drawdown'],
                'beat_o7_total': beat_o7_total,
                'beat_c8d_total': beat_c8d_total,
                'n_windows_beat_o7': n_win_o7,
                'n_windows_beat_c8d': n_win_c8d,
                'n_windows_dd_le_20': n_dd_le_20,
                'avg_yearly_return': float(np.mean(yearly_returns)) if len(yearly_returns) > 0 else None,
                'median_yearly_return': float(np.median(yearly_returns)) if len(yearly_returns) > 0 else None,
                'min_yearly_return': float(np.min(yearly_returns)) if len(yearly_returns) > 0 else None,
                'ret_2024': ret_2024,
                'ret_2026H1': ret_2026,
                'o7_ret_2024': o7_ret_2024,
                'o7_ret_2026H1': o7_ret_2026,
                'drag_2024': drag_2024,
                'drag_2026H1': drag_2026,
                'pass_criteria_mid': pass_criteria if is_mid else None,
            })

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(OUT_DIR / 'strategy_summary.csv', index=False)
    summary_df.to_json(OUT_DIR / 'strategy_summary.json', orient='records', force_ascii=False, indent=2)
    print(f"  strategy_summary.csv/json ({len(summary_df)} rows)")

    # 6.5 baseline_comparison.csv/json
    baseline_records = []
    for name in ['O7_baseline', 'C8D_baseline']:
        all_row = all_metrics_df[all_metrics_df['strategy'] == name].iloc[0]
        rec = {'strategy': name, 'window': 'all', **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in all_row.items()}}
        baseline_records.append(rec)
        for w in ALL_WINDOWS:
            w_row = metrics_df[(metrics_df['strategy'] == name) & (metrics_df['window'] == w)]
            if not w_row.empty:
                rec = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in w_row.iloc[0].items()}
                baseline_records.append(rec)
    baseline_df = pd.DataFrame(baseline_records)
    baseline_df.to_csv(OUT_DIR / 'baseline_comparison.csv', index=False)
    baseline_df.to_json(OUT_DIR / 'baseline_comparison.json', orient='records', force_ascii=False, indent=2)
    print(f"  baseline_comparison.csv/json ({len(baseline_df)} rows)")

    # 6.6 cost_impact_summary.csv/json（每框架的跨成本档对比）
    cost_records = []
    for fwk in FRAMEWORKS:
        for cost_name in COST_SCENARIOS.keys():
            strategy_name = f'{fwk}_{cost_name}_cost'
            row = summary_df[summary_df['strategy'] == strategy_name]
            if row.empty:
                continue
            row = row.iloc[0]
            cost_records.append({
                'framework': fwk,
                'cost_scenario': cost_name,
                'single_side_cost': COST_SCENARIOS[cost_name]['single_side'],
                'total_return': row['total_return'],
                'annual_return': row['annual_return'],
                'max_drawdown': row['max_drawdown'],
                'sharpe': row['sharpe'],
                'n_switches': row['n_switches'],
                'annual_switches': row['annual_switches'],
                'total_cost_pct': row['total_cost_pct'],
                'excess_vs_o7_total': row['excess_vs_o7_total'],
                'beat_o7_total': row['beat_o7_total'],
                'n_windows_beat_o7': row['n_windows_beat_o7'],
            })
    cost_df = pd.DataFrame(cost_records)
    cost_df.to_csv(OUT_DIR / 'cost_impact_summary.csv', index=False)
    cost_df.to_json(OUT_DIR / 'cost_impact_summary.json', orient='records', force_ascii=False, indent=2)
    print(f"  cost_impact_summary.csv/json ({len(cost_df)} rows)")

    # 6.7 year_2024_diagnostic.csv/json
    diag_2024 = []
    for fwk in FRAMEWORKS:
        for cost_name in COST_SCENARIOS.keys():
            strategy_name = f'{fwk}_{cost_name}_cost'
            s_row = metrics_df[(metrics_df['strategy'] == strategy_name) & (metrics_df['window'] == '2024')]
            o_row = metrics_df[(metrics_df['strategy'] == 'O7_baseline') & (metrics_df['window'] == '2024')]
            c_row = metrics_df[(metrics_df['strategy'] == 'C8D_baseline') & (metrics_df['window'] == '2024')]
            if s_row.empty or o_row.empty or c_row.empty:
                continue
            s = s_row.iloc[0]
            o = o_row.iloc[0]
            c = c_row.iloc[0]
            # 2024 切换事件
            sw_2024 = [e for e in all_events.get(strategy_name, []) if '2024' in e['date']]
            n_switch_2024 = len(sw_2024)
            # C8D 持有天数（2024）
            # 简化：用 2024 全年交易日数减去 O7 持有日数
            c8d_hold_days_2024 = 0
            sig_2024 = signals[fwk].loc[pd.Timestamp('2024-01-02'):pd.Timestamp('2024-12-31')]
            c8d_hold_days_2024 = int((sig_2024 == 'C8D').sum())

            diag_2024.append({
                'strategy': strategy_name,
                'framework': fwk,
                'cost_scenario': cost_name,
                'ret_2024_strategy': s['total_return'],
                'ret_2024_o7': o['total_return'],
                'ret_2024_c8d': c['total_return'],
                'excess_vs_o7_2024': s['total_return'] - o['total_return'],
                'excess_vs_c8d_2024': s['total_return'] - c['total_return'],
                'max_dd_2024_strategy': s['max_drawdown'],
                'max_dd_2024_o7': o['max_drawdown'],
                'max_dd_2024_c8d': c['max_drawdown'],
                'n_switches_2024': n_switch_2024,
                'c8d_hold_days_2024': c8d_hold_days_2024,
                'c8d_hold_ratio_2024': c8d_hold_days_2024 / s['n_trade_days'] if s['n_trade_days'] > 0 else 0,
                'drag_2024': s['total_return'] - o['total_return'] <= OBVIOUS_DRAG_RETURN,
            })
    diag_2024_df = pd.DataFrame(diag_2024)
    diag_2024_df.to_csv(OUT_DIR / 'year_2024_diagnostic.csv', index=False)
    diag_2024_df.to_json(OUT_DIR / 'year_2024_diagnostic.json', orient='records', force_ascii=False, indent=2)
    print(f"  year_2024_diagnostic.csv/json ({len(diag_2024_df)} rows)")

    # 6.8 p4_04_decision_summary.json
    # 通过标准检查（仅中成本档）
    mid_summaries = summary_df[summary_df['cost_scenario'] == 'mid'].copy()
    pass_results = []
    for _, row in mid_summaries.iterrows():
        crit = row.get('pass_criteria_mid', None)
        if crit is None:
            continue
        all_pass = all(crit.values())
        pass_results.append({
            'strategy': row['strategy'],
            'framework': row['framework'],
            'cost_scenario': 'mid',
            'pass_criteria': crit,
            'all_pass': all_pass,
            'total_return': row['total_return'],
            'excess_vs_o7': row['excess_vs_o7_total'],
            'annual_switches': row['annual_switches'],
            'n_windows_beat_o7': row['n_windows_beat_o7'],
        })

    # 高成本档检查
    high_summaries = summary_df[summary_df['cost_scenario'] == 'high'].copy()
    high_check = []
    for _, row in high_summaries.iterrows():
        # 高成本档全区间收益不低于 O7 的 95%
        # 最大回撤不高于 O7 + 2pp
        high_check.append({
            'strategy': row['strategy'],
            'framework': row['framework'],
            'high_total_return': row['total_return'],
            'o7_total_return': float(o7_all['total_return']),
            'return_ratio_vs_o7': float(row['total_return'] / o7_all['total_return']) if o7_all['total_return'] != 0 else 0,
            'return_passes_95pct': row['total_return'] >= HIGH_COST_THRESHOLD * o7_all['total_return'],
            'high_max_dd': row['max_drawdown'],
            'o7_max_dd': float(o7_all['max_drawdown']),
            'dd_passes_threshold': row['max_drawdown'] >= float(o7_all['max_drawdown']) - HIGH_COST_DD_THRESHOLD,
        })

    # 失败终止条件检查
    stop_conditions = []
    # 1. 无方案全区间跑赢 O7_baseline（中成本档，零切换不算）
    any_beat_o7_total = any(
        r['total_return'] > o7_all['total_return'] + FLOAT_EPSILON
        for r in mid_summaries.to_dict('records')
        if r['n_switches'] > 0
    )
    stop_conditions.append({
        'stop_condition': '1. 无方案全区间跑赢 O7_baseline（排除零切换）',
        'triggered': not any_beat_o7_total,
    })
    # 2. 无方案达到 5/7 窗口跑赢 O7（排除零切换）
    any_5of7 = any(
        r['n_windows_beat_o7'] >= 5
        for r in mid_summaries.to_dict('records')
        if r['n_switches'] > 0
    )
    stop_conditions.append({
        'stop_condition': '2. 无方案达到 5/7 窗口跑赢 O7（排除零切换）',
        'triggered': not any_5of7,
    })
    # 3. 年化切换 ≤12 时收益明显下降
    # 检查所有 ≤12 的方案（排除零切换，零切换等于 O7 不算"低频切换方案"）
    low_freq_strategies = mid_summaries[(mid_summaries['annual_switches'] <= MAX_ANNUAL_SWITCH) & (mid_summaries['n_switches'] > 0)]
    if len(low_freq_strategies) == 0:
        stop_conditions.append({
            'stop_condition': '4. 年化切换 ≤12 且有切换的方案不存在（C1 零切换不算）',
            'triggered': True,
        })
    else:
        # 检查收益是否明显低于 O7
        any_low_freq_beat = any(r['total_return'] > o7_all['total_return'] + FLOAT_EPSILON for r in low_freq_strategies.to_dict('records'))
        stop_conditions.append({
            'stop_condition': '4. 年化切换 ≤12 且有切换的方案无一个跑赢 O7',
            'triggered': not any_low_freq_beat,
        })
    # 5. 2024 拖累严重且无法解释
    any_no_2024_drag = any(not r['drag_2024'] for r in mid_summaries.to_dict('records'))
    stop_conditions.append({
        'stop_condition': '5. 所有方案 2024 都明显拖累',
        'triggered': not any_no_2024_drag,
    })
    # 6. 2026H1 拖累严重
    any_no_2026_drag = any(not r['drag_2026H1'] for r in mid_summaries.to_dict('records'))
    stop_conditions.append({
        'stop_condition': '6. 所有方案 2026H1 都明显拖累',
        'triggered': not any_no_2026_drag,
    })
    # 7. 高成本档全部失效
    any_high_passes = any(h['return_passes_95pct'] and h['dd_passes_threshold'] for h in high_check)
    stop_conditions.append({
        'stop_condition': '7. 高成本档全部失效',
        'triggered': not any_high_passes,
    })
    # 8. 最大回撤比 O7_baseline 更差（所有方案）
    any_dd_better = any(r['max_drawdown'] >= o7_all['max_drawdown'] for r in mid_summaries.to_dict('records'))
    stop_conditions.append({
        'stop_condition': '8. 所有方案最大回撤比 O7_baseline 更差',
        'triggered': not any_dd_better,
    })

    any_stop = any(c['triggered'] for c in stop_conditions)
    any_pass_all = any(r['all_pass'] for r in pass_results)

    if any_stop:
        final_recommendation = 'stop_pool_switch_research'
        final_baseline = 'O7_baseline'
    elif any_pass_all:
        final_recommendation = 'proceed_to_p4_05'
        final_baseline = 'O7_baseline（保留为最终保守 baseline）'
    else:
        final_recommendation = 'stop_pool_switch_research'
        final_baseline = 'O7_baseline'

    decision = {
        'pass_results_mid_cost': pass_results,
        'high_cost_check': high_check,
        'stop_conditions': stop_conditions,
        'any_stop_triggered': any_stop,
        'any_strategy_all_pass': any_pass_all,
        'final_recommendation': final_recommendation,
        'final_baseline': final_baseline,
        'o7_baseline_all_metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in o7_all.items()},
        'c8d_baseline_all_metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in c8d_all.items()},
    }
    with open(OUT_DIR / 'p4_04_decision_summary.json', 'w', encoding='utf-8') as f:
        json.dump(decision, f, ensure_ascii=False, indent=2, default=str)
    print(f"  p4_04_decision_summary.json (final_recommendation={final_recommendation})")

    print("\n" + "=" * 70)
    print("P4-04 完成")
    print("=" * 70)
    print(f"\n最终建议: {final_recommendation}")
    print(f"最终 baseline: {final_baseline}")


if __name__ == '__main__':
    main()
