"""
七星高照 P4-02 非动量变量组合预测力验证：O7 vs C8D
====================================================
验证目标：是否存在交易前可观察的非动量组合信号，能稳定预测未来一段时间
C8D 相对 O7 的占优/落后。

预测目标：target_excess_Nd = future_return(C8D_baseline, N) - future_return(O7_baseline, N)
- target_excess_Nd > 0 表示未来 N 日 C8D 优于 O7
- target_excess_Nd < 0 表示未来 N 日 O7 优于 C8D

未来收益计算口径（P3-02 修正版）：
  roll = (1 + daily_return).rolling(N).apply(np.prod, raw=True) - 1
  future_return_Nd = roll.shift(-(N - 1))   # T 到 T+N-1 的未来 N 日累计收益

主窗口 N=20，辅助窗口 N=5 / N=60。

边界：不生成切换规则、不给阈值、不回测切换策略、不调参、不新增池、不使用动量类主变量、
      不用未来收益反推方向、不用单一年份拟合、不宣布实盘、不修改 P1/P2/P3/P4-01 冻结结论。
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
P4_01_DIR = ROOT / 'qixing_optimize' / 'runs' / 'p4_01_core_pool_validation'
OUT_DIR = ROOT / 'qixing_optimize' / 'runs' / 'p4_02_non_momentum_predictive_validation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==== 池定义（P4-01 冻结） ====
POOL_O7 = [
    "518880.XSHG",
    "159985.XSHE",
    "501018.XSHG",
    "161226.XSHE",
    "513100.XSHG",
    "159915.XSHE",
    "511220.XSHG",
]
POOL_C8D = [
    "518880.XSHG",
    "159915.XSHE",
    "510300.XSHG",
    "510500.XSHG",
    "510050.XSHG",
    "513100.XSHG",
    "511220.XSHG",
    "159985.XSHE",
]
# O7 ∪ C8D = 9 只 ETF（518880/159915/513100/511220/159985 共享，501018/161226 仅 O7，510300/510500/510050 仅 C8D）
UNION_POOL = list(dict.fromkeys(POOL_O7 + POOL_C8D))

# ==== 7 个 walk-forward 窗口（P2/P4-01 冻结） ====
WINDOW_RANGES = OrderedDict([
    ('2020',   ('2020-01-02', '2020-12-31')),
    ('2021',   ('2021-01-04', '2021-12-31')),
    ('2022',   ('2022-01-04', '2022-12-30')),
    ('2023',   ('2023-01-03', '2023-12-29')),
    ('2024',   ('2024-01-02', '2024-12-31')),
    ('2025',   ('2025-01-02', '2025-12-31')),
    ('2026H1', ('2026-01-02', '2026-06-30')),
])

# ==== 预测窗口 ====
N_DAYS_LIST = [5, 20, 60]
PRIMARY_N = 20


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
    """日期 → 窗口标签"""
    d = pd.Timestamp(d)
    for w, (s, e) in WINDOW_RANGES.items():
        if pd.Timestamp(s) <= d <= pd.Timestamp(e):
            return w
    return None


# ============================================================
# 1. 加载 HData 日线数据
# ============================================================
def load_daily_data():
    """从 HData 1d_etf_price 读取 O7+C8D 并集 (9只) 的日线数据"""
    etf_dir = HDATA_ROOT / 'data' / 'raw' / '指数与ETF数据' / '1d_etf_price'
    code_map = {jq: _convert_code(jq) for jq in UNION_POOL}
    hd_codes = set(code_map.values())

    all_dfs = []
    for yr in range(2020, 2027):
        zp = etf_dir / f'{yr}.zip'
        if not zp.exists():
            print(f"  [跳过] {zp.name} 不存在")
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
                    yr_dfs.append(df[['code', 'date', 'close', 'high', 'low', 'vol', 'amount', 'adj_factor']])
        if yr_dfs:
            yr_df = pd.concat(yr_dfs, ignore_index=True)
            all_dfs.append(yr_df)
            print(f"  [加载] {yr}: {len(names)} 日, 筛选后 {len(yr_df)} 行")

    raw = pd.concat(all_dfs, ignore_index=True)
    raw['date'] = pd.to_datetime(raw['date'], format='%Y%m%d')

    # 前复权（与 P3-03 一致：latest_adj = 末日复权因子）
    latest_adj = raw.groupby('code')['adj_factor'].last()
    for col in ['close', 'high', 'low']:
        raw[f'fq_{col}'] = raw.apply(lambda r: r[col] * r['adj_factor'] / latest_adj[r['code']], axis=1)

    rev_map = {v: k for k, v in code_map.items()}

    def pivot(col):
        p = raw.pivot(index='date', columns='code', values=col).sort_index()
        return p.rename(columns=rev_map)

    result = {
        'close': pivot('fq_close'),
        'high': pivot('fq_high'),
        'low': pivot('fq_low'),
        'amount': pivot('amount'),
        'vol': pivot('vol'),
    }
    print(f"[HDATA] 价格矩阵: {result['close'].shape[0]} 交易日 × {result['close'].shape[1]} ETF")
    return result


# ============================================================
# 2. 加载 P4-01 日频净值（O7_baseline / C8D_baseline）
# ============================================================
def load_p4_01_daily_nav(pool_code='O7', score_mode='baseline'):
    """从 P4-01 回测 JSON 读取日频净值，拼接 7 个窗口"""
    navs = []
    for window in WINDOW_RANGES.keys():
        path = P4_01_DIR / f'{pool_code}_{score_mode}_{window}.json'
        if not path.exists():
            print(f"  [警告] 缺失: {path.name}")
            continue
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        daily = d['daily']
        for r in daily:
            navs.append({'date': r['date'], 'value': r['value'], 'window': window})
    df = pd.DataFrame(navs)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    dup = df[df['date'].duplicated(keep=False)]
    if len(dup) > 0:
        print(f"  [警告] {pool_code}_{score_mode} 有 {len(dup)} 个重复日期，保留首个")
    df = df.drop_duplicates('date').reset_index(drop=True)
    return df


# ============================================================
# 3. 构建目标面板（target_panel）
# ============================================================
def build_target_panel(o7_nav, c8d_nav):
    """构建预测目标：target_excess_Nd = future(C8D,N) - future(O7,N)

    未来 N 日收益口径（P3-02 修正版）：
      roll = (1 + r).rolling(N).apply(np.prod) - 1   # 在 i+N-1 处算 [i, i+N-1]
      future_return_Nd = roll.shift(-(N-1))           # 把 i+N-1 的值移到 i，得到 [i, i+N-1]
    """
    df = pd.DataFrame({
        'o7_value': o7_nav.set_index('date')['value'],
        'c8d_value': c8d_nav.set_index('date')['value'],
    }).dropna()
    df['o7_ret'] = df['o7_value'].pct_change()
    df['c8d_ret'] = df['c8d_value'].pct_change()

    targets = pd.DataFrame(index=df.index)
    targets['o7_value'] = df['o7_value']
    targets['c8d_value'] = df['c8d_value']
    targets['o7_ret'] = df['o7_ret']
    targets['c8d_ret'] = df['c8d_ret']
    targets['window'] = [date_to_window(d) for d in targets.index]

    for n in N_DAYS_LIST:
        # Future N-day return (T to T+N-1)
        roll_o7 = (1 + df['o7_ret']).rolling(n).apply(np.prod, raw=True) - 1
        roll_c8d = (1 + df['c8d_ret']).rolling(n).apply(np.prod, raw=True) - 1
        targets[f'future_o7_{n}d'] = roll_o7.shift(-(n - 1))
        targets[f'future_c8d_{n}d'] = roll_c8d.shift(-(n - 1))
        targets[f'target_excess_{n}d'] = targets[f'future_c8d_{n}d'] - targets[f'future_o7_{n}d']
        targets[f'target_win_{n}d'] = (targets[f'target_excess_{n}d'] > 0).astype(int)

    return targets


# ============================================================
# 4. 构建特征面板（feature_panel）
# ============================================================
def build_feature_panel(data):
    """构建 6 类非动量特征变量（T 可计算，target 用未来收益，无前视偏差）"""
    close = data['close']
    amount = data['amount']

    # 日收益率（按 ETF）
    etf_rets = close.pct_change()

    # 池等权日收益（每日用可用 ETF 等权）
    def pool_eq_weight_ret(pool):
        sub = etf_rets[pool].dropna(how='all')
        n_valid = sub.notna().sum(axis=1)
        eq_ret = sub.sum(axis=1) / n_valid.replace(0, np.nan)
        return eq_ret.fillna(0)

    o7_pool_ret = pool_eq_weight_ret(POOL_O7)
    c8d_pool_ret = pool_eq_weight_ret(POOL_C8D)

    # 池净值（累积）用于回撤计算
    o7_pool_nav = (1 + o7_pool_ret).cumprod()
    c8d_pool_nav = (1 + c8d_pool_ret).cumprod()

    features = pd.DataFrame(index=close.index)

    # === A. 回撤位置类 ===
    def drawdown(nav_series, window):
        peak = nav_series.rolling(window, min_periods=1).max()
        return (nav_series - peak) / peak

    features['c8d_dd_20d'] = drawdown(c8d_pool_nav, 20)
    features['c8d_dd_60d'] = drawdown(c8d_pool_nav, 60)
    features['c8d_dd_120d'] = drawdown(c8d_pool_nav, 120)
    features['o7_dd_20d'] = drawdown(o7_pool_nav, 20)
    features['o7_dd_60d'] = drawdown(o7_pool_nav, 60)
    features['o7_dd_120d'] = drawdown(o7_pool_nav, 120)
    features['dd_diff_60d'] = features['c8d_dd_60d'] - features['o7_dd_60d']
    features['dd_diff_120d'] = features['c8d_dd_120d'] - features['o7_dd_120d']

    # === B. 波动率类 ===
    features['c8d_vol_20d'] = c8d_pool_ret.rolling(20).std()
    features['o7_vol_20d'] = o7_pool_ret.rolling(20).std()
    features['vol_ratio_20d'] = features['c8d_vol_20d'] / features['o7_vol_20d'].replace(0, np.nan)
    c8d_vol_60d = c8d_pool_ret.rolling(60).std()
    o7_vol_60d = o7_pool_ret.rolling(60).std()
    features['c8d_vol_expansion'] = features['c8d_vol_20d'] / c8d_vol_60d.replace(0, np.nan) - 1
    features['o7_vol_expansion'] = features['o7_vol_20d'] / o7_vol_60d.replace(0, np.nan) - 1
    features['vol_expansion_diff'] = features['c8d_vol_expansion'] - features['o7_vol_expansion']

    # === C. 横截面离散度类 ===
    def pool_dispersion(pool, window=20):
        sub = etf_rets[pool]
        daily_std = sub.std(axis=1)
        return daily_std.rolling(window).mean()

    features['c8d_internal_dispersion_20d'] = pool_dispersion(POOL_C8D, 20)
    features['o7_internal_dispersion_20d'] = pool_dispersion(POOL_O7, 20)
    features['dispersion_diff'] = features['c8d_internal_dispersion_20d'] - features['o7_internal_dispersion_20d']
    features['union_dispersion_20d'] = pool_dispersion(UNION_POOL, 20)

    # Top1 减去中位数（横截面，20日累积收益）
    def top1_minus_median(pool, window=20):
        sub = etf_rets[pool]
        cum_ret = (1 + sub).rolling(window).apply(np.prod, raw=True) - 1
        top1 = cum_ret.max(axis=1)
        med = cum_ret.median(axis=1)
        return top1 - med

    features['top1_minus_median_union_20d'] = top1_minus_median(UNION_POOL, 20)
    features['top1_minus_median_c8d_20d'] = top1_minus_median(POOL_C8D, 20)
    features['top1_minus_median_o7_20d'] = top1_minus_median(POOL_O7, 20)

    # === D. 相关性 / 拥挤度类 ===
    def pool_internal_corr(pool, window=60):
        sub = etf_rets[pool].dropna(how='all')
        n = len(pool)
        if n < 2:
            return pd.Series(np.nan, index=sub.index)
        corrs = []
        cols = [c for c in pool if c in sub.columns]
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = sub[cols[i]].rolling(window).corr(sub[cols[j]])
                corrs.append(c)
        if not corrs:
            return pd.Series(np.nan, index=sub.index)
        return pd.concat(corrs, axis=1).mean(axis=1)

    features['c8d_internal_corr_60d'] = pool_internal_corr(POOL_C8D, 60)
    features['o7_internal_corr_60d'] = pool_internal_corr(POOL_O7, 60)
    features['corr_diff_60d'] = features['c8d_internal_corr_60d'] - features['o7_internal_corr_60d']
    features['cross_pool_corr_60d'] = o7_pool_ret.rolling(60).corr(c8d_pool_ret)

    # === E. 趋势效率 / 噪音类 ===
    def efficiency(ret_series, window=20):
        roll_sum_abs = ret_series.abs().rolling(window).sum()
        roll_cum = (1 + ret_series).rolling(window).apply(np.prod, raw=True) - 1
        return roll_cum.abs() / roll_sum_abs.replace(0, np.nan)

    features['c8d_efficiency_20d'] = efficiency(c8d_pool_ret, 20)
    features['o7_efficiency_20d'] = efficiency(o7_pool_ret, 20)
    features['efficiency_diff'] = features['c8d_efficiency_20d'] - features['o7_efficiency_20d']

    features['c8d_up_day_ratio_20d'] = (c8d_pool_ret > 0).rolling(20).mean()
    features['o7_up_day_ratio_20d'] = (o7_pool_ret > 0).rolling(20).mean()
    features['up_day_ratio_diff'] = features['c8d_up_day_ratio_20d'] - features['o7_up_day_ratio_20d']

    def max_day_contrib(ret_series, window=20):
        roll_max_abs = ret_series.abs().rolling(window).max()
        roll_sum_abs = ret_series.abs().rolling(window).sum()
        return roll_max_abs / roll_sum_abs.replace(0, np.nan)

    features['c8d_max_day_contrib_20d'] = max_day_contrib(c8d_pool_ret, 20)
    features['o7_max_day_contrib_20d'] = max_day_contrib(o7_pool_ret, 20)
    features['max_day_contrib_diff'] = features['c8d_max_day_contrib_20d'] - features['o7_max_day_contrib_20d']

    # === F. 流动性类 ===
    def pool_amt_mean(pool, window=20):
        sub = amount[pool]
        avg = sub.mean(axis=1)
        return avg.rolling(window).mean()

    features['c8d_amt_mean_20d'] = pool_amt_mean(POOL_C8D, 20)
    features['o7_amt_mean_20d'] = pool_amt_mean(POOL_O7, 20)
    features['amt_ratio_20d'] = features['c8d_amt_mean_20d'] / features['o7_amt_mean_20d'].replace(0, np.nan)

    c8d_amt_60d = pool_amt_mean(POOL_C8D, 60)
    o7_amt_60d = pool_amt_mean(POOL_O7, 60)
    features['c8d_amt_change'] = features['c8d_amt_mean_20d'] / c8d_amt_60d.replace(0, np.nan) - 1
    features['o7_amt_change'] = features['o7_amt_mean_20d'] / o7_amt_60d.replace(0, np.nan) - 1
    features['amt_change_diff'] = features['c8d_amt_change'] - features['o7_amt_change']

    return features


# ============================================================
# 5. 构建组合分数
# ============================================================
def build_combo_scores(features):
    """构建两个固定组合分数（不调权重，等权 z-score 加总）"""
    def z(s):
        m = s.mean()
        sd = s.std()
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=s.index)
        return (s - m) / sd

    # 方法 2：MR score（均值回归倾向）
    # 方向假设（来自 P3-03 观察）：
    #   -C8D 深回撤更有利（c8d_dd_120d 越负，未来修复概率越高）
    #   -C8D 成交额收缩更有利（拥挤度下降）
    #   -C8D 波动率收缩更有利（恐慌期结束）
    #   -Top1 领先过大更不利（头部拥挤后回归）
    #   +C8D 路径效率高更有利（趋势更纯净）
    mr_score = (
        z(-features['c8d_dd_120d'])
        + z(-features['c8d_amt_change'])
        + z(-features['c8d_vol_expansion'])
        + z(-features['top1_minus_median_union_20d'])
        + z(features['c8d_efficiency_20d'])
    )

    # 方法 3：relative_score（O7 vs C8D 相对状态）
    # 方向假设：
    #   +dd_diff_120d（C8D 相对 O7 回撤更深时，C8D 更可能修复）
    #   -vol_expansion_diff（C8D 波动扩张小于 O7 时，C8D 更稳）
    #   -amt_change_diff（C8D 成交额扩张小于 O7 时，C8D 更稳）
    #   +efficiency_diff（C8D 路径效率高于 O7 时，C8D 更优）
    #   -dispersion_diff（C8D 内部分化小于 O7 时，C8D 更稳）
    relative_score = (
        z(features['dd_diff_120d'])
        + z(-features['vol_expansion_diff'])
        + z(-features['amt_change_diff'])
        + z(features['efficiency_diff'])
        + z(-features['dispersion_diff'])
    )

    return pd.DataFrame({
        'mr_score': mr_score,
        'relative_score': relative_score,
    })


# ============================================================
# 6. 单变量 / 组合检验
# ============================================================
def predictive_test(feature_series, target_series, feature_name, n_buckets=3):
    """单变量预测力检验：IC, rank IC, high-low spread, hit rate, by year

    target_excess_Nd > 0 表示 C8D 优于 O7。
    IC > 0 表示特征值越大，C8D 越可能占优。
    """
    df = pd.DataFrame({
        'feature': feature_series,
        'target': target_series,
    }).dropna()

    if len(df) < 30:
        return None

    # 全样本
    ic = df['feature'].corr(df['target'])
    rank_ic = df['feature'].corr(df['target'], method='spearman')

    # 全样本分桶
    try:
        df['bucket'] = pd.qcut(df['feature'], n_buckets, labels=['low', 'mid', 'high'])
    except Exception:
        return None

    bucket_means = df.groupby('bucket')['target'].mean()
    bucket_counts = df.groupby('bucket')['target'].count()
    high_mean = float(bucket_means.get('high', np.nan))
    low_mean = float(bucket_means.get('low', np.nan))
    high_low_spread = high_mean - low_mean

    # Hit rate：高桶中 target 方向与 IC 方向一致的比例
    high_data = df[df['bucket'] == 'high']
    low_data = df[df['bucket'] == 'low']
    if not np.isnan(ic) and ic > 0:
        # IC > 0：高桶应更多 target > 0
        hit_rate = float((high_data['target'] > 0).mean()) if len(high_data) > 0 else np.nan
    elif not np.isnan(ic) and ic < 0:
        # IC < 0：高桶应更多 target < 0
        hit_rate = float((high_data['target'] < 0).mean()) if len(high_data) > 0 else np.nan
    else:
        hit_rate = np.nan

    # 按年份拆开（用全样本分位阈值的 cut，保证方向可比）
    df['window'] = [date_to_window(d) for d in df.index]
    q_low = df['feature'].quantile(1 / n_buckets)
    q_high = df['feature'].quantile(2 / n_buckets)

    by_year = {}
    for w in WINDOW_RANGES.keys():
        wsub = df[df['window'] == w].copy()
        if len(wsub) < 10:
            continue
        w_ic = wsub['feature'].corr(wsub['target'])
        w_rank_ic = wsub['feature'].corr(wsub['target'], method='spearman')
        try:
            wsub['bucket'] = pd.cut(wsub['feature'],
                                    bins=[-np.inf, q_low, q_high, np.inf],
                                    labels=['low', 'mid', 'high'])
            w_bucket_means = wsub.groupby('bucket')['target'].mean()
            w_high_low = float(w_bucket_means.get('high', np.nan) - w_bucket_means.get('low', np.nan))
        except Exception:
            w_high_low = np.nan
        by_year[w] = {
            'ic': round(float(w_ic), 4) if not np.isnan(w_ic) else None,
            'rank_ic': round(float(w_rank_ic), 4) if not np.isnan(w_rank_ic) else None,
            'high_low_spread': round(w_high_low, 4) if not np.isnan(w_high_low) else None,
            'n': int(len(wsub)),
        }

    # 方向一致性：IC 符号在多少年份成立
    ic_sign = np.sign(ic) if not np.isnan(ic) else 0
    consistent_years = 0
    total_years = 0
    for v in by_year.values():
        if v['ic'] is not None:
            total_years += 1
            if np.sign(v['ic']) == ic_sign and ic_sign != 0:
                consistent_years += 1

    # 2026H1 检查
    y2026 = by_year.get('2026H1', {})
    y2026_correct = False
    if y2026.get('ic') is not None and ic_sign != 0:
        y2026_correct = (np.sign(y2026['ic']) == ic_sign)

    return {
        'feature': feature_name,
        'n_total': int(len(df)),
        'ic': round(float(ic), 4) if not np.isnan(ic) else None,
        'rank_ic': round(float(rank_ic), 4) if not np.isnan(rank_ic) else None,
        'high_mean': round(high_mean, 4) if not np.isnan(high_mean) else None,
        'low_mean': round(low_mean, 4) if not np.isnan(low_mean) else None,
        'mid_mean': round(float(bucket_means.get('mid', np.nan)), 4) if not np.isnan(bucket_means.get('mid', np.nan)) else None,
        'high_low_spread': round(float(high_low_spread), 4) if not np.isnan(high_low_spread) else None,
        'high_count': int(bucket_counts.get('high', 0)),
        'low_count': int(bucket_counts.get('low', 0)),
        'mid_count': int(bucket_counts.get('mid', 0)),
        'hit_rate': round(float(hit_rate), 4) if not np.isnan(hit_rate) else None,
        'by_year': by_year,
        'consistent_years': consistent_years,
        'total_years': total_years,
        'direction_consistent_5of7': consistent_years >= 5,
        'y2026_correct_direction': y2026_correct,
    }


# ============================================================
# 7. 主流程
# ============================================================
def main():
    print(f"{'=' * 80}")
    print(f"P4-02 非动量变量组合预测力验证：O7 vs C8D")
    print(f"{'=' * 80}")

    # 1. 加载 HData 日线数据
    print(f"\n[1/8] 加载 HData 日线数据 (O7∪C8D = {len(UNION_POOL)} 只 ETF)...")
    data = load_daily_data()

    # 2. 加载 P4-01 日频净值
    print(f"\n[2/8] 加载 P4-01 日频净值...")
    o7_nav = load_p4_01_daily_nav('O7', 'baseline')
    c8d_nav = load_p4_01_daily_nav('C8D', 'baseline')
    print(f"  O7_baseline: {len(o7_nav)} 交易日, 范围 {o7_nav['date'].min().date()} ~ {o7_nav['date'].max().date()}")
    print(f"  C8D_baseline: {len(c8d_nav)} 交易日, 范围 {c8d_nav['date'].min().date()} ~ {c8d_nav['date'].max().date()}")

    # 3. 构建目标面板
    print(f"\n[3/8] 构建目标面板 (target_excess_Nd = future(C8D,N) - future(O7,N))...")
    targets = build_target_panel(o7_nav, c8d_nav)
    print(f"  目标面板: {targets.shape[0]} 交易日, {targets.shape[1]} 列")
    print(f"  主目标 target_excess_{PRIMARY_N}d: 均值={targets[f'target_excess_{PRIMARY_N}d'].mean():.4f}, "
          f"标准差={targets[f'target_excess_{PRIMARY_N}d'].std():.4f}")
    win_dist = targets['window'].value_counts().sort_index()
    print(f"  窗口分布: {dict(win_dist)}")

    # 4. 构建特征面板
    print(f"\n[4/8] 构建特征面板 (6 类非动量变量)...")
    features = build_feature_panel(data)
    # 对齐到 target 的日期
    common_dates = sorted(set(features.index) & set(targets.index))
    features = features.loc[common_dates]
    targets = targets.loc[common_dates]
    print(f"  特征面板: {features.shape[0]} 交易日 × {features.shape[1]} 特征")
    print(f"  对齐后交易日数: {len(common_dates)}")

    # 数据审计
    audit = {
        'n_total_days': len(common_dates),
        'n_features': features.shape[1],
        'n_targets': targets.shape[1],
        'feature_coverage': {
            col: {
                'n_valid': int(features[col].notna().sum()),
                'coverage_pct': round(float(features[col].notna().mean() * 100), 2),
            } for col in features.columns
        },
        'target_coverage': {
            col: {
                'n_valid': int(targets[col].notna().sum()),
                'coverage_pct': round(float(targets[col].notna().mean() * 100), 2),
            } for col in targets.columns if col not in ('window',)
        },
        'window_distribution': {w: int((targets['window'] == w).sum()) for w in WINDOW_RANGES.keys()},
        'future_return_method': 'roll.shift(-(N-1)) -> T to T+N-1 累计收益 (P3-02 修正版)',
        'pool_o7': POOL_O7,
        'pool_c8d': POOL_C8D,
        'n_days_list': N_DAYS_LIST,
        'primary_n': PRIMARY_N,
    }

    # 保存 feature_panel / target_panel
    features.to_csv(OUT_DIR / 'feature_panel.csv')
    targets.to_csv(OUT_DIR / 'target_panel.csv')
    features.reset_index().to_json(OUT_DIR / 'feature_panel.json', orient='records', date_format='iso')
    targets.reset_index().to_json(OUT_DIR / 'target_panel.json', orient='records', date_format='iso')

    # 5. 单变量检验（主窗口 N=20，并检验 N=5/60）
    print(f"\n[5/8] 单变量预测力检验 (N=5/20/60)...")
    univariate_results = {}
    for n in N_DAYS_LIST:
        target_col = f'target_excess_{n}d'
        print(f"\n  --- N={n} ---")
        for feat_name in features.columns:
            res = predictive_test(features[feat_name], targets[target_col], feat_name)
            if res is not None:
                res['n_days'] = n
                res['target_col'] = target_col
                univariate_results[f'{feat_name}@N{n}'] = res
                if n == PRIMARY_N:
                    flag = '✓5/7' if res['direction_consistent_5of7'] else ' '
                    print(f"    {flag} {feat_name:<35} IC={res['ic']:>7}  H-L={res['high_low_spread']:>7}  "
                          f"一致 {res['consistent_years']}/{res['total_years']}  2026H1={'✓' if res['y2026_correct_direction'] else '✗'}")

    # 6. 组合变量检验
    print(f"\n[6/8] 组合变量检验 (MR score + relative score)...")
    combo_scores = build_combo_scores(features)
    combo_scores = combo_scores.loc[common_dates]
    combo_results = {}
    for n in N_DAYS_LIST:
        target_col = f'target_excess_{n}d'
        print(f"\n  --- N={n} ---")
        for combo_name in combo_scores.columns:
            res = predictive_test(combo_scores[combo_name], targets[target_col], combo_name)
            if res is not None:
                res['n_days'] = n
                res['target_col'] = target_col
                res['combo_method'] = 'mr_score' if combo_name == 'mr_score' else 'relative_score'
                combo_results[f'{combo_name}@N{n}'] = res
                flag = '✓5/7' if res['direction_consistent_5of7'] else ' '
                print(f"    {flag} {combo_name:<20} IC={res['ic']:>7}  H-L={res['high_low_spread']:>7}  "
                      f"一致 {res['consistent_years']}/{res['total_years']}  2026H1={'✓' if res['y2026_correct_direction'] else '✗'}")

    # 7. 年度稳定性矩阵
    print(f"\n[7/8] 年度稳定性矩阵...")
    stability_rows = []
    for key, res in {**univariate_results, **combo_results}.items():
        if res['n_days'] != PRIMARY_N:
            continue
        row = {'feature': res['feature'], 'n_days': res['n_days']}
        for w in WINDOW_RANGES.keys():
            yr = res['by_year'].get(w, {})
            row[f'ic_{w}'] = yr.get('ic')
            row[f'hl_{w}'] = yr.get('high_low_spread')
        row['consistent_years'] = res['consistent_years']
        row['total_years'] = res['total_years']
        row['direction_consistent_5of7'] = res['direction_consistent_5of7']
        row['y2026_correct'] = res['y2026_correct_direction']
        row['ic_full'] = res['ic']
        row['rank_ic_full'] = res['rank_ic']
        row['high_low_spread_full'] = res['high_low_spread']
        stability_rows.append(row)
    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(OUT_DIR / 'yearly_stability_matrix.csv', index=False)
    stability_df.to_json(OUT_DIR / 'yearly_stability_matrix.json', orient='records', indent=2)

    # 8. 保存所有结果
    print(f"\n[8/8] 保存结果...")

    # univariate_results
    uni_rows = []
    for key, res in univariate_results.items():
        row = {k: v for k, v in res.items() if k != 'by_year'}
        row['by_year'] = res['by_year']
        uni_rows.append(row)
    pd.DataFrame(uni_rows).to_csv(OUT_DIR / 'univariate_results.csv', index=False)
    with open(OUT_DIR / 'univariate_results.json', 'w', encoding='utf-8') as f:
        json.dump(univariate_results, f, indent=2, ensure_ascii=False, default=str)

    # combo_results
    combo_rows = []
    for key, res in combo_results.items():
        row = {k: v for k, v in res.items() if k != 'by_year'}
        row['by_year'] = res['by_year']
        combo_rows.append(row)
    pd.DataFrame(combo_rows).to_csv(OUT_DIR / 'combo_results.csv', index=False)
    with open(OUT_DIR / 'combo_results.json', 'w', encoding='utf-8') as f:
        json.dump(combo_results, f, indent=2, ensure_ascii=False, default=str)

    # data audit
    with open(OUT_DIR / 'data_audit.json', 'w', encoding='utf-8') as f:
        json.dump(audit, f, indent=2, ensure_ascii=False, default=str)

    # === 最终汇总打印 ===
    print(f"\n{'=' * 80}")
    print(f"P4-02 单变量 + 组合变量主窗口 N={PRIMARY_N} 汇总")
    print(f"{'=' * 80}")
    print(f"{'变量':<40} {'IC':>8} {'rankIC':>8} {'H-L':>8} {'一致':>6} {'5/7':>5} {'2026H1':>7}")
    print(f"{'-' * 80}")
    for key, res in {**univariate_results, **combo_results}.items():
        if res['n_days'] != PRIMARY_N:
            continue
        flag57 = '✓' if res['direction_consistent_5of7'] else ' '
        flag26 = '✓' if res['y2026_correct_direction'] else '✗'
        print(f"{res['feature']:<40} {str(res['ic']):>8} {str(res['rank_ic']):>8} "
              f"{str(res['high_low_spread']):>8} {res['consistent_years']}/{res['total_years']:>3} "
              f"{flag57:>5} {flag26:>7}")

    # 通过标准检查
    print(f"\n{'=' * 80}")
    print(f"通过标准检查（5/7 方向一致 + 2026H1 不严重反向 + H-L 有经济意义）")
    print(f"{'=' * 80}")
    pass_count = 0
    for key, res in {**univariate_results, **combo_results}.items():
        if res['n_days'] != PRIMARY_N:
            continue
        passed = (
            res['direction_consistent_5of7']
            and res['y2026_correct_direction']
            and res['high_low_spread'] is not None
            and abs(res['high_low_spread']) > 0.005  # 经济意义门槛：H-L spread > 0.5%
        )
        if passed:
            pass_count += 1
            print(f"  [通过] {res['feature']}: 一致 {res['consistent_years']}/{res['total_years']}, "
                  f"H-L={res['high_low_spread']}, 2026H1={'✓' if res['y2026_correct_direction'] else '✗'}")
    if pass_count == 0:
        print(f"  [无变量通过] 所有变量均未达到 5/7 + 2026H1 正确 + H-L>0.5% 的标准")
    else:
        print(f"\n  共 {pass_count} 个变量通过通过标准")

    print(f"\n输出目录: {OUT_DIR}")
    print(f"输出文件:")
    for f in sorted(OUT_DIR.glob('*')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
