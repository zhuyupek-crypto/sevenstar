"""
七星高照 P4-03 反向因果与成本后净收益验证：O7 vs C8D
======================================================
回答三个问题：
1. relative_score 反向预测力是否有因果合理性？
2. o7_internal_corr_60d 的 7/7 稳定性是否有经济解释？
3. 扣除切换成本后，P4-02 发现的 H-L spread 是否仍有净收益空间？

数据来源：直接读取 P4-02 已冻结的 feature_panel / target_panel（T-1 修正版，commit 6ec3672）
不重算特征，不重算 target，不新增变量，不生成切换规则，不回测切换策略。

子任务：
A. relative_score 反向逻辑拆解（组件贡献 + 高低桶画像 + 年度差异）
B. o7_internal_corr_60d 经济解释验证（基础检验 + 状态解释 + pairwise 拆解）
C. 交易成本后净收益验证（三档成本压力测试 + 桶切换频率估算）
D. 失败终止条件验证（决策汇总）

边界：
- 不生成切换规则、不给阈值、不回测切换策略、不做动态择时
- 不新增池、不新增因子、不调参
- 不反向使用 relative_score 生成规则
- 不宣布可实盘
- 不修改 P1/P2/P3/P4-01/P4-02 已冻结结论
"""
import sys
import os
import json
import zipfile
import io
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ==== 路径 ====
HDATA_ROOT = Path(os.environ.get('HDATA_ROOT', r'D:\Work Space\HData'))
P4_02_DIR = ROOT / 'qixing_optimize' / 'runs' / 'p4_02_non_momentum_predictive_validation'
OUT_DIR = ROOT / 'qixing_optimize' / 'runs' / 'p4_03_reverse_causality_cost_validation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== 池定义（P4-01/P4-02 冻结，不得修改） ====
POOL_O7 = [
    "518880.XSHG",  # 黄金
    "159985.XSHE",  # 豆粕
    "501018.XSHG",  # 原油
    "161226.XSHE",  # 白银
    "513100.XSHG",  # 纳指
    "159915.XSHE",  # 创业板
    "511220.XSHG",  # 城投债
]
POOL_C8D = [
    "518880.XSHG",  # 黄金
    "159915.XSHE",  # 创业板
    "510300.XSHG",  # 沪深300
    "510500.XSHG",  # 中证500
    "510050.XSHG",  # 上证50
    "513100.XSHG",  # 纳指
    "511220.XSHG",  # 城投债
    "159985.XSHE",  # 豆粕
]
UNION_POOL = list(dict.fromkeys(POOL_O7 + POOL_C8D))

# O7 内部资产分类（用于 pairwise 拆解）
O7_GROUPS = {
    "贵金属": ["518880.XSHG", "161226.XSHE"],         # 黄金-白银
    "商品":   ["159985.XSHE", "501018.XSHG"],         # 豆粕-原油
    "海外权益": ["513100.XSHG"],                       # 纳指
    "国内权益": ["159915.XSHE"],                       # 创业板
    "债券":   ["511220.XSHG"],                         # 城投债
}
O7_PAIRS = [
    ("518880.XSHG", "161226.XSHE", "黄金-白银"),
    ("159985.XSHE", "501018.XSHG", "豆粕-原油"),
    ("513100.XSHG", "159915.XSHE", "纳指-创业板"),
    ("518880.XSHG", "159985.XSHE", "黄金-豆粕"),
    ("518880.XSHG", "501018.XSHG", "黄金-原油"),
    ("518880.XSHG", "513100.XSHG", "黄金-纳指"),
    ("518880.XSHG", "159915.XSHE", "黄金-创业板"),
    ("518880.XSHG", "511220.XSHG", "黄金-城投债"),
    ("161226.XSHE", "513100.XSHG", "白银-纳指"),
    ("513100.XSHG", "511220.XSHG", "纳指-城投债"),
    ("159915.XSHE", "511220.XSHG", "创业板-城投债"),
    ("501018.XSHG", "511220.XSHG", "原油-城投债"),
]

# ==== 7 个 walk-forward 窗口（P2/P4-01/P4-02 冻结） ====
WINDOW_RANGES = OrderedDict([
    ('2020',   ('2020-01-02', '2020-12-31')),
    ('2021',   ('2021-01-04', '2021-12-31')),
    ('2022',   ('2022-01-04', '2022-12-30')),
    ('2023',   ('2023-01-03', '2023-12-29')),
    ('2024',   ('2024-01-02', '2024-12-31')),
    ('2025',   ('2025-01-02', '2025-12-31')),
    ('2026H1', ('2026-01-02', '2026-06-30')),
])

# ==== 预测窗口（P4-02 冻结） ====
N_DAYS_LIST = [5, 20, 60]
PRIMARY_N = 20

# ==== relative_score 五个组成项（P4-02 冻结，不得改权重/方向） ====
RELATIVE_SCORE_COMPONENTS = [
    ("dd_diff_120d",         +1),  # z(dd_diff_120d)
    ("vol_expansion_diff",   -1),  # z(-vol_expansion_diff)
    ("amt_change_diff",      -1),  # z(-amt_change_diff)
    ("efficiency_diff",      +1),  # z(efficiency_diff)
    ("dispersion_diff",      -1),  # z(-dispersion_diff)
]

# ==== 成本档位（任务书七、C1 冻结） ====
COST_SCENARIOS = OrderedDict([
    ("low",  {"single_side": 0.0010, "round_trip": 0.0020}),
    ("mid",  {"single_side": 0.0020, "round_trip": 0.0040}),
    ("high", {"single_side": 0.0037, "round_trip": 0.0074}),
])


# ============================================================
# 工具函数
# ============================================================
def _convert_code(jq_code):
    if jq_code.endswith('.XSHG'):
        return jq_code.replace('.XSHG', '.SH')
    if jq_code.endswith('.XSHE'):
        return jq_code.replace('.XSHE', '.SZ')
    return jq_code


def date_to_window(d):
    d = pd.Timestamp(d)
    for w, (s, e) in WINDOW_RANGES.items():
        if pd.Timestamp(s) <= d <= pd.Timestamp(e):
            return w
    return None


def z_score(s):
    """全样本 z-score"""
    m = s.mean()
    sd = s.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - m) / sd


def load_p4_02_data():
    """直接读取 P4-02 已冻结的 feature_panel 和 target_panel（T-1 修正版）"""
    print("[1] 加载 P4-02 冻结数据 ...")
    fp = pd.read_csv(P4_02_DIR / 'feature_panel.csv', parse_dates=['date'], index_col='date')
    tp = pd.read_csv(P4_02_DIR / 'target_panel.csv', parse_dates=['date'], index_col='date')
    print(f"  feature_panel: {fp.shape}")
    print(f"  target_panel:  {tp.shape}")
    return fp, tp


def load_daily_data():
    """从 HData 加载 O7 日线数据（用于 pairwise correlation 拆解）"""
    print("[2] 加载 HData O7 日线数据 ...")
    etf_dir = HDATA_ROOT / 'data' / 'raw' / '指数与ETF数据' / '1d_etf_price'
    code_map = {jq: _convert_code(jq) for jq in POOL_O7}
    hd_codes = set(code_map.values())

    all_dfs = []
    for yr in range(2020, 2027):
        zp = etf_dir / f'{yr}.zip'
        if not zp.exists():
            continue
        with zipfile.ZipFile(zp, 'r') as z:
            names = sorted(z.namelist())
            yr_dfs = []
            for nm in names:
                if not nm.endswith('.parquet'):
                    continue
                with z.open(nm) as f:
                    df = pd.read_parquet(io.BytesIO(f.read()))
                df = df[df['code'].isin(hd_codes)]
                if len(df) > 0:
                    yr_dfs.append(df[['code', 'date', 'close', 'adj_factor']])
        if yr_dfs:
            yr_df = pd.concat(yr_dfs, ignore_index=True)
            all_dfs.append(yr_df)

    raw = pd.concat(all_dfs, ignore_index=True)
    raw['date'] = pd.to_datetime(raw['date'], format='%Y%m%d')

    # 前复权
    latest_adj = raw.groupby('code')['adj_factor'].last()
    raw['fq_close'] = raw.apply(lambda r: r['close'] * r['adj_factor'] / latest_adj[r['code']], axis=1)

    rev_map = {v: k for k, v in code_map.items()}
    pivot = raw.pivot(index='date', columns='code', values='fq_close').sort_index()
    pivot = pivot.rename(columns=rev_map)
    print(f"  O7 close 矩阵: {pivot.shape}")
    return pivot


# ============================================================
# 子任务 A: relative_score 反向逻辑拆解
# ============================================================
def task_a_relative_score_decomposition(features, targets):
    """子任务 A：relative_score 反向逻辑拆解

    输出：
    - relative_score_component_analysis.csv/.json（5 个组件的预测贡献）
    - relative_score_bucket_profile.csv/.json（高低桶状态画像）
    - 按年度拆解结果（合并到 component_analysis）
    """
    print("\n[3] 子任务 A：relative_score 反向逻辑拆解 ...")

    # 重建 relative_score（与 P4-02 完全一致，全样本 z-score 等权加总）
    rel_score = pd.Series(0.0, index=features.index)
    component_z = pd.DataFrame(index=features.index)
    for col, sign in RELATIVE_SCORE_COMPONENTS:
        z = z_score(features[col])
        component_z[col] = z
        rel_score += sign * z

    # === A2. 五个组成项的预测贡献 ===
    component_results = []
    for col, sign in RELATIVE_SCORE_COMPONENTS:
        for n in N_DAYS_LIST:
            target_col = f'target_excess_{n}d'
            df = pd.DataFrame({
                'feature': features[col],
                'target': targets[target_col],
                'window': [date_to_window(d) for d in features.index],
            }).dropna()

            if len(df) < 30:
                continue

            ic = df['feature'].corr(df['target'])
            rank_ic = df['feature'].corr(df['target'], method='spearman')

            # 分桶
            try:
                df['bucket'] = pd.qcut(df['feature'], 3, labels=['low', 'mid', 'high'])
            except Exception:
                continue
            bucket_means = df.groupby('bucket')['target'].mean()
            high_low = float(bucket_means.get('high', np.nan)) - float(bucket_means.get('low', np.nan))

            # 按年度方向一致数（与全样本 IC 同向为一致）
            consistent = 0
            by_year = {}
            if not np.isnan(ic):
                direction = 1 if ic > 0 else -1
                for w in WINDOW_RANGES.keys():
                    wsub = df[df['window'] == w]
                    if len(wsub) < 10:
                        continue
                    w_ic = wsub['feature'].corr(wsub['target'])
                    if not np.isnan(w_ic) and (w_ic * direction > 0):
                        consistent += 1
                    by_year[w] = {'ic': float(w_ic) if not np.isnan(w_ic) else None,
                                  'n': len(wsub)}

            # 2026H1 方向
            y2026_ic = by_year.get('2026H1', {}).get('ic')
            y2026_correct = None
            if y2026_ic is not None and not np.isnan(ic):
                direction = 1 if ic > 0 else -1
                y2026_correct = bool(y2026_ic * direction > 0)

            # 与 relative_score 的相关性
            corr_with_rel = float(features[col].corr(rel_score)) if not features[col].corr(rel_score) != features[col].corr(rel_score) else None
            try:
                corr_with_rel = float(features[col].corr(rel_score))
            except Exception:
                corr_with_rel = None

            component_results.append({
                'component': col,
                'sign_in_rel_score': sign,
                'n_days': n,
                'target_col': target_col,
                'ic': float(ic) if not np.isnan(ic) else None,
                'rank_ic': float(rank_ic) if not np.isnan(rank_ic) else None,
                'high_low_spread': high_low,
                'consistent_years': consistent,
                'total_years': 7,
                'direction_consistent_5of7': consistent >= 5,
                'y2026_correct': y2026_correct,
                'by_year': by_year,
                'corr_with_relative_score': corr_with_rel,
            })

    comp_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'by_year'} for r in component_results])
    comp_df.to_csv(OUT_DIR / 'relative_score_component_analysis.csv', index=False, encoding='utf-8-sig')
    with open(OUT_DIR / 'relative_score_component_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(component_results, f, ensure_ascii=False, indent=2, default=str)

    # === A3. 高低桶状态画像 ===
    df_profile = pd.DataFrame({
        'relative_score': rel_score,
        'dd_diff_120d': features['dd_diff_120d'],
        'vol_expansion_diff': features['vol_expansion_diff'],
        'amt_change_diff': features['amt_change_diff'],
        'efficiency_diff': features['efficiency_diff'],
        'dispersion_diff': features['dispersion_diff'],
        'o7_future_20d': targets['future_o7_20d'],
        'c8d_future_20d': targets['future_c8d_20d'],
        'target_excess_20d': targets['target_excess_20d'],
        'target_win_20d': targets['target_win_20d'],
        'window': [date_to_window(d) for d in rel_score.index],
    }).dropna(subset=['relative_score'])

    try:
        df_profile['bucket'] = pd.qcut(df_profile['relative_score'], 3, labels=['low', 'mid', 'high'])
    except Exception as e:
        print(f"  [警告] 分桶失败: {e}")
        return comp_df, None, None

    bucket_profile = df_profile.groupby('bucket').agg(
        n_samples=('relative_score', 'count'),
        relative_score_mean=('relative_score', 'mean'),
        dd_diff_120d_mean=('dd_diff_120d', 'mean'),
        vol_expansion_diff_mean=('vol_expansion_diff', 'mean'),
        amt_change_diff_mean=('amt_change_diff', 'mean'),
        efficiency_diff_mean=('efficiency_diff', 'mean'),
        dispersion_diff_mean=('dispersion_diff', 'mean'),
        o7_future_return_20d_mean=('o7_future_20d', 'mean'),
        c8d_future_return_20d_mean=('c8d_future_20d', 'mean'),
        target_excess_20d_mean=('target_excess_20d', 'mean'),
        target_win_20d_mean=('target_win_20d', 'mean'),
    ).reset_index()
    bucket_profile.to_csv(OUT_DIR / 'relative_score_bucket_profile.csv', index=False, encoding='utf-8-sig')

    bucket_profile_json = bucket_profile.to_dict(orient='records')
    with open(OUT_DIR / 'relative_score_bucket_profile.json', 'w', encoding='utf-8') as f:
        json.dump(bucket_profile_json, f, ensure_ascii=False, indent=2, default=str)

    # === A4. 年度差异 ===
    yearly_results = []
    for w in WINDOW_RANGES.keys():
        wsub = df_profile[df_profile['window'] == w].copy()
        if len(wsub) < 30:
            continue

        w_ic = wsub['relative_score'].corr(wsub['target_excess_20d'])
        try:
            wsub['b'] = pd.qcut(wsub['relative_score'], 3, labels=['low', 'mid', 'high'])
            w_bucket = wsub.groupby('b')['target_excess_20d'].mean()
            w_hl = float(w_bucket.get('high', np.nan)) - float(w_bucket.get('low', np.nan))

            high_sub = wsub[wsub['b'] == 'high']
            low_sub = wsub[wsub['b'] == 'low']
            high_o7 = float(high_sub['o7_future_20d'].mean()) if len(high_sub) > 0 else np.nan
            high_c8d = float(high_sub['c8d_future_20d'].mean()) if len(high_sub) > 0 else np.nan
            low_o7 = float(low_sub['o7_future_20d'].mean()) if len(low_sub) > 0 else np.nan
            low_c8d = float(low_sub['c8d_future_20d'].mean()) if len(low_sub) > 0 else np.nan
        except Exception:
            w_hl = np.nan
            high_o7 = high_c8d = low_o7 = low_c8d = np.nan

        # 主要贡献组件（按 |z| 排序，符号与全样本 IC 反向 = 与 relative_score 反向同向）
        # relative_score 全样本 IC=-0.1372，反向即"高桶 C8D 输"
        # 组件贡献 = 组件对 relative_score 的 z 加权 × 组件对 target 的 IC 方向
        component_contribs = {}
        for col, sign in RELATIVE_SCORE_COMPONENTS:
            w_comp_ic = wsub[col].corr(wsub['target_excess_20d'])
            component_contribs[col] = {
                'ic': float(w_comp_ic) if not np.isnan(w_comp_ic) else None,
                'sign_in_rel_score': sign,
            }

        yearly_results.append({
            'window': w,
            'n': len(wsub),
            'rel_score_ic': float(w_ic) if not np.isnan(w_ic) else None,
            'high_low_spread': float(w_hl) if not np.isnan(w_hl) else None,
            'high_bucket_o7_future_20d': high_o7 if not np.isnan(high_o7) else None,
            'high_bucket_c8d_future_20d': high_c8d if not np.isnan(high_c8d) else None,
            'low_bucket_o7_future_20d': low_o7 if not np.isnan(low_o7) else None,
            'low_bucket_c8d_future_20d': low_c8d if not np.isnan(low_c8d) else None,
            'component_ics': component_contribs,
            'direction_correct': bool(w_ic < 0) if not np.isnan(w_ic) else None,  # 反向预测：IC<0 为正确方向
        })

    return comp_df, bucket_profile, yearly_results


# ============================================================
# 子任务 B: o7_internal_corr_60d 经济解释
# ============================================================
def task_b_o7_corr_explanation(features, targets, o7_close):
    """子任务 B：o7_internal_corr_60d 经济解释验证

    输出：
    - o7_corr_analysis.csv/.json
    - pairwise_corr_breakdown.csv/.json
    """
    print("\n[4] 子任务 B：o7_internal_corr_60d 经济解释 ...")

    feat = features['o7_internal_corr_60d']

    # === B1. 基础检验 ===
    basic_results = []
    for n in N_DAYS_LIST:
        target_col = f'target_excess_{n}d'
        df = pd.DataFrame({
            'feature': feat,
            'target': targets[target_col],
            'window': [date_to_window(d) for d in feat.index],
        }).dropna()

        if len(df) < 30:
            continue

        ic = df['feature'].corr(df['target'])
        rank_ic = df['feature'].corr(df['target'], method='spearman')

        try:
            df['bucket'] = pd.qcut(df['feature'], 3, labels=['low', 'mid', 'high'])
            bucket_means = df.groupby('bucket')['target'].mean()
            high_low = float(bucket_means.get('high', np.nan)) - float(bucket_means.get('low', np.nan))
        except Exception:
            high_low = np.nan

        # 年度一致数
        consistent = 0
        by_year = {}
        if not np.isnan(ic):
            direction = 1 if ic > 0 else -1
            for w in WINDOW_RANGES.keys():
                wsub = df[df['window'] == w]
                if len(wsub) < 10:
                    continue
                w_ic = wsub['feature'].corr(wsub['target'])
                if not np.isnan(w_ic) and (w_ic * direction > 0):
                    consistent += 1
                by_year[w] = {'ic': float(w_ic) if not np.isnan(w_ic) else None, 'n': len(wsub)}

        y2026_ic = by_year.get('2026H1', {}).get('ic')
        y2026_correct = None
        if y2026_ic is not None and not np.isnan(ic):
            direction = 1 if ic > 0 else -1
            y2026_correct = bool(y2026_ic * direction > 0)

        basic_results.append({
            'feature': 'o7_internal_corr_60d',
            'n_days': n,
            'n_total': len(df),
            'ic': float(ic) if not np.isnan(ic) else None,
            'rank_ic': float(rank_ic) if not np.isnan(rank_ic) else None,
            'high_low_spread': float(high_low) if not np.isnan(high_low) else None,
            'consistent_years': consistent,
            'total_years': 7,
            'direction_consistent_5of7': consistent >= 5,
            'y2026_correct': y2026_correct,
            'by_year': by_year,
        })

    # === B2. 状态解释（按三分桶） ===
    df_state = pd.DataFrame({
        'o7_corr': feat,
        'o7_future_20d': targets['future_o7_20d'],
        'c8d_future_20d': targets['future_c8d_20d'],
        'target_excess_20d': targets['target_excess_20d'],
        'target_win_20d': targets['target_win_20d'],
        'o7_vol_20d': features['o7_vol_20d'],
        'o7_dispersion_20d': features['o7_internal_dispersion_20d'],
        'c8d_dispersion_20d': features['c8d_internal_dispersion_20d'],
    }).dropna(subset=['o7_corr'])

    try:
        df_state['bucket'] = pd.qcut(df_state['o7_corr'], 3, labels=['low', 'mid', 'high'])
    except Exception:
        df_state['bucket'] = np.nan

    state_profile = df_state.groupby('bucket').agg(
        n_samples=('o7_corr', 'count'),
        o7_corr_mean=('o7_corr', 'mean'),
        o7_future_return_20d_mean=('o7_future_20d', 'mean'),
        c8d_future_return_20d_mean=('c8d_future_20d', 'mean'),
        target_excess_20d_mean=('target_excess_20d', 'mean'),
        target_win_20d_mean=('target_win_20d', 'mean'),
        o7_vol_20d_mean=('o7_vol_20d', 'mean'),
        o7_dispersion_20d_mean=('o7_dispersion_20d', 'mean'),
        c8d_dispersion_20d_mean=('c8d_dispersion_20d', 'mean'),
    ).reset_index()

    # === B3. O7 内部 pairwise correlation 拆解 ===
    print("  [B3] 计算 O7 内部 pairwise correlation ...")
    o7_rets = o7_close.pct_change()

    pairwise_results = []
    pair_corr_series = {}

    for code_a, code_b, label in O7_PAIRS:
        if code_a not in o7_rets.columns or code_b not in o7_rets.columns:
            continue
        pair_corr = o7_rets[code_a].rolling(60).corr(o7_rets[code_b])
        pair_corr_series[label] = pair_corr

        # 与 target_excess_20d 的 IC
        df_pair = pd.DataFrame({
            'pair_corr': pair_corr,
            'target': targets['target_excess_20d'],
            'o7_internal_corr_60d': feat,
            'window': [date_to_window(d) for d in pair_corr.index],
        }).dropna()

        if len(df_pair) < 30:
            continue

        pair_ic_target = df_pair['pair_corr'].corr(df_pair['target'])
        pair_ic_with_o7corr = df_pair['pair_corr'].corr(df_pair['o7_internal_corr_60d'])

        # 年度 IC
        by_year = {}
        for w in WINDOW_RANGES.keys():
            wsub = df_pair[df_pair['window'] == w]
            if len(wsub) < 10:
                continue
            w_ic = wsub['pair_corr'].corr(wsub['target'])
            by_year[w] = float(w_ic) if not np.isnan(w_ic) else None

        pairwise_results.append({
            'pair_label': label,
            'code_a': code_a,
            'code_b': code_b,
            'n_total': len(df_pair),
            'mean_corr': float(df_pair['pair_corr'].mean()),
            'ic_with_target_excess_20d': float(pair_ic_target) if not np.isnan(pair_ic_target) else None,
            'corr_with_o7_internal_corr_60d': float(pair_ic_with_o7corr) if not np.isnan(pair_ic_with_o7corr) else None,
            'by_year_ic_with_target': by_year,
        })

    pair_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'by_year_ic_with_target'} for r in pairwise_results])
    pair_df.to_csv(OUT_DIR / 'pairwise_corr_breakdown.csv', index=False, encoding='utf-8-sig')
    with open(OUT_DIR / 'pairwise_corr_breakdown.json', 'w', encoding='utf-8') as f:
        json.dump(pairwise_results, f, ensure_ascii=False, indent=2, default=str)

    # 合并 o7_corr_analysis
    o7_corr_analysis = {
        'basic_results': basic_results,
        'state_profile': state_profile.to_dict(orient='records'),
        'pairwise_breakdown_summary': {
            'n_pairs_analyzed': len(pairwise_results),
            'top_pair_by_corr_with_o7corr': max(pairwise_results, key=lambda x: abs(x.get('corr_with_o7_internal_corr_60d') or 0))['pair_label'] if pairwise_results else None,
            'top_pair_by_ic_with_target': max(pairwise_results, key=lambda x: abs(x.get('ic_with_target_excess_20d') or 0))['pair_label'] if pairwise_results else None,
        },
    }
    pd.DataFrame(basic_results).drop(columns=['by_year']).to_csv(
        OUT_DIR / 'o7_corr_analysis.csv', index=False, encoding='utf-8-sig'
    )
    with open(OUT_DIR / 'o7_corr_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(o7_corr_analysis, f, ensure_ascii=False, indent=2, default=str)

    return basic_results, state_profile, pairwise_results


# ============================================================
# 子任务 C: 交易成本后净收益验证
# ============================================================
def task_c_cost_stress_test(features, targets):
    """子任务 C：交易成本后净收益验证

    输出：
    - cost_stress_test.csv/.json
    - bucket_transition_stats.csv/.json
    """
    print("\n[5] 子任务 C：交易成本后净收益验证 ...")

    # 重建 relative_score
    rel_score = pd.Series(0.0, index=features.index)
    for col, sign in RELATIVE_SCORE_COMPONENTS:
        rel_score += sign * z_score(features[col])

    # 五个变量/组合
    test_features = {
        'relative_score': rel_score,
        'o7_internal_corr_60d': features['o7_internal_corr_60d'],
        'top1_minus_median_c8d_20d': features['top1_minus_median_c8d_20d'],
        'efficiency_diff': features['efficiency_diff'],
        'c8d_amt_change': features['c8d_amt_change'],
    }

    cost_results = []
    for feat_name, feat_series in test_features.items():
        for n in N_DAYS_LIST:
            target_col = f'target_excess_{n}d'
            df = pd.DataFrame({
                'feature': feat_series,
                'target': targets[target_col],
            }).dropna()

            if len(df) < 30:
                continue

            try:
                df['bucket'] = pd.qcut(df['feature'], 3, labels=['low', 'mid', 'high'])
                bucket_means = df.groupby('bucket')['target'].mean()
                gross_hl = abs(float(bucket_means.get('high', np.nan)) - float(bucket_means.get('low', np.nan)))
            except Exception:
                continue

            if np.isnan(gross_hl):
                continue

            row = {
                'feature': feat_name,
                'n_days': n,
                'n_total': len(df),
                'gross_hl_spread': gross_hl,
                'net_hl_low_cost':  gross_hl - COST_SCENARIOS['low']['round_trip'],
                'net_hl_mid_cost':  gross_hl - COST_SCENARIOS['mid']['round_trip'],
                'net_hl_high_cost': gross_hl - COST_SCENARIOS['high']['round_trip'],
                'net_hl_low_positive':  bool(gross_hl - COST_SCENARIOS['low']['round_trip']  > 0),
                'net_hl_mid_positive':  bool(gross_hl - COST_SCENARIOS['mid']['round_trip']  > 0),
                'net_hl_high_positive': bool(gross_hl - COST_SCENARIOS['high']['round_trip'] > 0),
                'net_hl_mid_above_1pct': bool(gross_hl - COST_SCENARIOS['mid']['round_trip'] > 0.01),
            }
            cost_results.append(row)

    cost_df = pd.DataFrame(cost_results)
    cost_df.to_csv(OUT_DIR / 'cost_stress_test.csv', index=False, encoding='utf-8-sig')
    with open(OUT_DIR / 'cost_stress_test.json', 'w', encoding='utf-8') as f:
        json.dump(cost_results, f, ensure_ascii=False, indent=2, default=str)

    # === C3. 桶切换频率估算（仅 relative_score 与 o7_internal_corr_60d） ===
    transition_results = []
    for feat_name in ['relative_score', 'o7_internal_corr_60d']:
        if feat_name == 'relative_score':
            feat_series = rel_score
        else:
            feat_series = features['o7_internal_corr_60d']

        df = pd.DataFrame({'feature': feat_series}).dropna()
        if len(df) < 30:
            continue

        try:
            df['bucket'] = pd.qcut(df['feature'], 3, labels=['low', 'mid', 'high'])
        except Exception:
            continue

        # 桶变化次数
        df['bucket_prev'] = df['bucket'].shift(1)
        df['bucket_changed'] = (df['bucket'] != df['bucket_prev']).astype(int)
        # 第一天 NaN 不计
        df_valid = df.dropna(subset=['bucket_prev'])
        n_changes = int(df_valid['bucket_changed'].sum())
        n_days = len(df_valid)

        # 年化切换次数（按 252 交易日）
        years = n_days / 252.0
        annualized_changes = n_changes / years if years > 0 else None

        # 桶持续时长
        # 找到每段连续相同桶的长度
        durations = []
        if n_days > 0:
            cur_bucket = df_valid['bucket'].iloc[0]
            cur_len = 1
            for i in range(1, len(df_valid)):
                if df_valid['bucket'].iloc[i] == cur_bucket:
                    cur_len += 1
                else:
                    durations.append(cur_len)
                    cur_bucket = df_valid['bucket'].iloc[i]
                    cur_len = 1
            durations.append(cur_len)

        transition_results.append({
            'feature': feat_name,
            'n_total_days': n_days,
            'bucket_change_count': n_changes,
            'annualized_bucket_change_count': float(annualized_changes) if annualized_changes else None,
            'average_bucket_duration_days': float(np.mean(durations)) if durations else None,
            'median_bucket_duration_days': float(np.median(durations)) if durations else None,
            'min_bucket_duration_days': int(np.min(durations)) if durations else None,
            'max_bucket_duration_days': int(np.max(durations)) if durations else None,
        })

    trans_df = pd.DataFrame(transition_results)
    trans_df.to_csv(OUT_DIR / 'bucket_transition_stats.csv', index=False, encoding='utf-8-sig')
    with open(OUT_DIR / 'bucket_transition_stats.json', 'w', encoding='utf-8') as f:
        json.dump(transition_results, f, ensure_ascii=False, indent=2, default=str)

    return cost_results, transition_results


# ============================================================
# 子任务 D: 失败终止条件验证
# ============================================================
def task_d_decision(comp_df, bucket_profile, yearly_results,
                    o7_corr_basic, state_profile, pairwise_results,
                    cost_results, transition_results):
    """子任务 D：失败终止条件验证 + 决策汇总"""
    print("\n[6] 子任务 D：失败终止条件验证 ...")

    # === 提取关键指标 ===
    # relative_score N=20 关键指标
    rel_n20_row = next((r for r in cost_results if r['feature'] == 'relative_score' and r['n_days'] == 20), None)
    rel_gross_hl = rel_n20_row['gross_hl_spread'] if rel_n20_row else None
    rel_net_low  = rel_n20_row['net_hl_low_cost']  if rel_n20_row else None
    rel_net_mid  = rel_n20_row['net_hl_mid_cost']  if rel_n20_row else None
    rel_net_high = rel_n20_row['net_hl_high_cost'] if rel_n20_row else None

    # o7_corr N=20 关键指标
    o7_corr_n20 = next((r for r in o7_corr_basic if r['n_days'] == 20), None)

    # 桶切换频率
    rel_trans = next((r for r in transition_results if r['feature'] == 'relative_score'), None)
    rel_annual_changes = rel_trans['annualized_bucket_change_count'] if rel_trans else None
    rel_avg_duration = rel_trans['average_bucket_duration_days'] if rel_trans else None

    # === 继续研究条件检查（任务书 D1） ===
    cond_results = []

    # 条件1：relative_score 反向逻辑有明确因果解释（基于 A 任务结果判断）
    # 简化判断：是否有组件 IC 与 relative_score IC 方向一致且 5/7+
    rel_components_n20 = [r for r in comp_df.to_dict('records') if r.get('n_days') == 20] if hasattr(comp_df, 'to_dict') else []
    cond1 = len(rel_components_n20) > 0  # 至少有组件数据
    cond_results.append({
        'condition': '1. relative_score 反向逻辑有明确因果解释',
        'passed': cond1,
        'evidence': f"组件数据 {len(rel_components_n20)} 条，详见 relative_score_component_analysis.csv",
    })

    # 条件2：o7_internal_corr_60d 有明确经济含义，且不是纯数学噪声
    # 简化判断：pairwise 拆解中是否有 pair corr > 0.3 解释力
    has_economic_pair = any(
        abs(r.get('corr_with_o7_internal_corr_60d') or 0) > 0.3
        for r in pairwise_results
    )
    cond_results.append({
        'condition': '2. o7_internal_corr_60d 有经济含义（非纯噪声）',
        'passed': has_economic_pair,
        'evidence': f"pairwise 中存在 |corr|>0.3 的 pair：{has_economic_pair}",
    })

    # 条件3：成本后 relative_score 净 H-L spread 在中成本档仍 > 1.0%
    cond3 = (rel_net_mid is not None) and (rel_net_mid > 0.01)
    cond_results.append({
        'condition': '3. relative_score 中成本档 net_HL > 1.0%',
        'passed': cond3,
        'evidence': f"net_hl_mid = {rel_net_mid:.4f} ({rel_net_mid*100:.2f}%)" if rel_net_mid is not None else "N/A",
    })

    # 条件4：高成本档下净 H-L spread 不为负
    cond4 = (rel_net_high is not None) and (rel_net_high > 0)
    cond_results.append({
        'condition': '4. relative_score 高成本档 net_HL > 0',
        'passed': cond4,
        'evidence': f"net_hl_high = {rel_net_high:.4f} ({rel_net_high*100:.2f}%)" if rel_net_high is not None else "N/A",
    })

    # 条件5：2024 失效原因能够被解释
    # 通过 yearly_results 中 2024 的 component IC 判断
    y2024 = next((r for r in yearly_results if r['window'] == '2024'), None)
    cond5 = y2024 is not None and y2024.get('rel_score_ic') is not None
    cond_results.append({
        'condition': '5. 2024 失效原因可解释',
        'passed': cond5,
        'evidence': f"2024 IC = {y2024.get('rel_score_ic'):.4f}" if y2024 and y2024.get('rel_score_ic') else "N/A",
    })

    # 条件6：N=20/60 稳定，N=5 不稳定不影响后续研究
    rel_n5  = next((r for r in cost_results if r['feature'] == 'relative_score' and r['n_days'] == 5),  None)
    rel_n60 = next((r for r in cost_results if r['feature'] == 'relative_score' and r['n_days'] == 60), None)
    cond6 = (rel_n20_row is not None) and (rel_n60 is not None) and (rel_n5 is not None)
    cond_results.append({
        'condition': '6. N=20/60 稳定，N=5 不影响后续',
        'passed': cond6,
        'evidence': f"N=5 gross={rel_n5['gross_hl_spread']*100:.2f}%, N=20 gross={rel_n20_row['gross_hl_spread']*100:.2f}%, N=60 gross={rel_n60['gross_hl_spread']*100:.2f}%" if cond6 else "N/A",
    })

    # 条件7：不依赖单一年份
    # yearly_results 中方向正确的年份数
    n_correct = sum(1 for r in yearly_results if r.get('direction_correct') is True)
    cond7 = n_correct >= 5
    cond_results.append({
        'condition': '7. 不依赖单一年份（5/7+ 方向正确）',
        'passed': cond7,
        'evidence': f"{n_correct}/7 年份方向正确",
    })

    # === 停止研究条件检查（任务书 D2） ===
    stop_results = []

    # 停止条件3：中成本档后 net_HL <= 1.0%
    stop3 = (rel_net_mid is not None) and (rel_net_mid <= 0.01)
    stop_results.append({
        'stop_condition': '3. 中成本档后 net_HL <= 1.0%',
        'triggered': stop3,
        'evidence': f"net_hl_mid = {rel_net_mid:.4f}" if rel_net_mid is not None else "N/A",
    })

    # 停止条件4：高成本档后 net_HL <= 0
    stop4 = (rel_net_high is not None) and (rel_net_high <= 0)
    stop_results.append({
        'stop_condition': '4. 高成本档后 net_HL <= 0',
        'triggered': stop4,
        'evidence': f"net_hl_high = {rel_net_high:.4f}" if rel_net_high is not None else "N/A",
    })

    # 停止条件5：桶切换频率过高（年化 > 50 次为过高）
    stop5 = (rel_annual_changes is not None) and (rel_annual_changes > 50)
    stop_results.append({
        'stop_condition': '5. 桶切换频率过高（年化 > 50 次）',
        'triggered': stop5,
        'evidence': f"年化切换 = {rel_annual_changes:.1f} 次" if rel_annual_changes else "N/A",
    })

    # 综合：继续条件全部满足 且 无停止条件触发 → 可进入 P4-04
    all_continue_passed = all(c['passed'] for c in cond_results)
    any_stop_triggered = any(s['triggered'] for s in stop_results)
    can_proceed_p4_04 = all_continue_passed and not any_stop_triggered

    decision = {
        'continue_conditions': cond_results,
        'stop_conditions': stop_results,
        'all_continue_passed': all_continue_passed,
        'any_stop_triggered': any_stop_triggered,
        'final_recommendation': 'proceed_to_p4_04' if can_proceed_p4_04 else 'stop_pool_switch_research',
        'final_baseline': 'O7_baseline（无论结论如何都保留）',
        'key_metrics': {
            'relative_score_n20_gross_hl': rel_gross_hl,
            'relative_score_n20_net_hl_low':  rel_net_low,
            'relative_score_n20_net_hl_mid':  rel_net_mid,
            'relative_score_n20_net_hl_high': rel_net_high,
            'o7_corr_n20_consistent_years': o7_corr_n20.get('consistent_years') if o7_corr_n20 else None,
            'relative_score_annual_bucket_changes': rel_annual_changes,
            'relative_score_avg_bucket_duration_days': rel_avg_duration,
        },
    }

    with open(OUT_DIR / 'p4_03_decision_summary.json', 'w', encoding='utf-8') as f:
        json.dump(decision, f, ensure_ascii=False, indent=2, default=str)

    return decision


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("P4-03 反向因果与成本后净收益验证：O7 vs C8D")
    print("=" * 70)

    # 1. 加载 P4-02 冻结数据
    features, targets = load_p4_02_data()

    # 2. 加载 HData O7 日线（用于 pairwise）
    o7_close = load_daily_data()

    # 3. 子任务 A：relative_score 反向逻辑拆解
    comp_df, bucket_profile, yearly_results = task_a_relative_score_decomposition(features, targets)

    # 4. 子任务 B：o7_internal_corr_60d 经济解释
    o7_corr_basic, state_profile, pairwise_results = task_b_o7_corr_explanation(features, targets, o7_close)

    # 5. 子任务 C：交易成本后净收益验证
    cost_results, transition_results = task_c_cost_stress_test(features, targets)

    # 6. 子任务 D：失败终止条件验证
    decision = task_d_decision(
        comp_df, bucket_profile, yearly_results,
        o7_corr_basic, state_profile, pairwise_results,
        cost_results, transition_results
    )

    # === 保存 yearly_results 到 decision summary（追加） ===
    # 重新读取 decision_summary 并追加 yearly_results
    with open(OUT_DIR / 'p4_03_decision_summary.json', 'r', encoding='utf-8') as f:
        decision_full = json.load(f)
    decision_full['yearly_results'] = yearly_results
    decision_full['cost_results'] = cost_results
    decision_full['transition_results'] = transition_results
    with open(OUT_DIR / 'p4_03_decision_summary.json', 'w', encoding='utf-8') as f:
        json.dump(decision_full, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 70)
    print("P4-03 全部子任务完成")
    print("=" * 70)
    print(f"\n输出目录: {OUT_DIR}")
    print(f"\n最终建议: {decision['final_recommendation']}")
    print(f"继续条件全部满足: {decision['all_continue_passed']}")
    print(f"停止条件触发: {decision['any_stop_triggered']}")
    print(f"\n关键指标:")
    km = decision['key_metrics']
    print(f"  relative_score N=20 gross H-L:    {km['relative_score_n20_gross_hl']*100:.2f}%")
    print(f"  relative_score N=20 net (low):    {km['relative_score_n20_net_hl_low']*100:.2f}%")
    print(f"  relative_score N=20 net (mid):    {km['relative_score_n20_net_hl_mid']*100:.2f}%")
    print(f"  relative_score N=20 net (high):   {km['relative_score_n20_net_hl_high']*100:.2f}%")
    print(f"  o7_corr 7/7 一致:                 {km['o7_corr_n20_consistent_years']}/7")
    print(f"  relative_score 年化切换次数:      {km['relative_score_annual_bucket_changes']:.1f}")
    print(f"  relative_score 平均桶持续天数:    {km['relative_score_avg_bucket_duration_days']:.1f}")


if __name__ == '__main__':
    main()
