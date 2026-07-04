"""
P3-02 候选切换因子的因果论证
=========================================
输入：HData 日线收盘价（独立 MTM，不用 P3-01 残差法）
输出：factor_validation/
  - daily_portfolio_nav.json   原池/大池独有/主题等权组合日频净值
  - factor_series.json         4 个交易前因子的 T-1 日频序列
  - bucket_stats.json          因子分桶 vs 未来收益统计（全样本 + 7 年份）
  - posthoc_capital_usage.json 事后诊断变量（资金占用率，非交易前因子）
  - P3-02_FACTOR_VALIDATION_REPORT.md

严格边界（用户修订版）：
1. 4 个交易前因子（T-1 口径）+ 1 个事后诊断变量（资金占用率，单独一节）
2. 全部因子用 T-1 数据，观察 T~T+N 未来收益
3. 不给阈值，只输出"是否有解释力 + 高低分桶方向性差异 + 按年份稳定性"
4. 分桶统计按年份拆开（全样本 + 2020-2026H1）
5. HData 日线收盘价独立 MTM，不用残差法
6. 不生成规则，不调参，不回测切换策略
"""
import json
import math
import sys
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# ==== 路径 ====
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'qixing_optimize' / 'runs' / 'factor_validation'
OUT_DIR.mkdir(parents=True, exist_ok=True)
WF_DIR = ROOT / 'qixing_optimize' / 'runs' / 'walk_forward'
ATTR_DIR = ROOT / 'qixing_optimize' / 'runs' / 'attribution'

# ==== HData 环境 ====
os.environ.setdefault('LOCAL_QUANT_HDATA_SOURCE', 'core')
os.environ.setdefault('HDATA_ROOT', r'D:\Work Space\HData')
_LOCAL_QUANT = r'D:\Work Space\local_quant'
if _LOCAL_QUANT not in sys.path:
    sys.path.insert(0, _LOCAL_QUANT)
# local_quant 内部用 core 包，需要确保 core 可导入
_CORE_PATH = os.path.join(_LOCAL_QUANT, 'core')
if os.path.exists(_CORE_PATH):
    sys.path.insert(0, _LOCAL_QUANT)  # 让 from core import xxx 可用

# ==== 池定义（与 P3-01 一致） ====
DEFAULT_POOL = [
    '518880.XSHG', '159980.XSHE', '159985.XSHE', '501018.XSHG',
    '161226.XSHE', '513100.XSHG', '511220.XSHG', '511880.XSHG',
]
# 大池独有 29 只（从 P3-01 trades 反推，POOL_BAK - DEFAULT_POOL）
BAK_ONLY = [
    '159509.XSHE', '159529.XSHE', '159792.XSHE', '159920.XSHE', '159967.XSHE',
    '159981.XSHE', '510210.XSHG', '510300.XSHG', '510500.XSHG', '511010.XSHG',
    '511380.XSHG', '512040.XSHG', '512100.XSHG', '512890.XSHG', '513030.XSHG',
    '513050.XSHG', '513080.XSHG', '513130.XSHG', '513290.XSHG', '513310.XSHG',
    '513400.XSHG', '513500.XSHG', '513520.XSHG', '513690.XSHG', '513730.XSHG',
    '563300.XSHG', '563360.XSHG', '588080.XSHG',
]
# 注意：159980 在原池里（黄金），不在 BAK_ONLY
ALL_ETFS = DEFAULT_POOL + BAK_ONLY  # 8 + 28 = 36（159980 重复，去重）

# ==== 主题映射 ====
ETF_THEME = {
    '518880.XSHG': '黄金', '159980.XSHE': '黄金', '159985.XSHE': '黄金',
    '501018.XSHG': '原油', '159981.XSHE': '原油',
    '161226.XSHE': '白酒', '513690.XSHG': '白酒',
    '513100.XSHG': '纳指', '159509.XSHE': '纳指', '513290.XSHG': '纳指科技',
    '511220.XSHG': '十年国债', '511380.XSHG': '十年国债', '511010.XSHG': '国债',
    '511880.XSHG': '货币(防御)',
    '513500.XSHG': '中概互联', '159529.XSHE': '中概互联', '513050.XSHG': '中概互联',
    '513400.XSHG': '日经', '513520.XSHG': '日经',
    '513030.XSHG': '德国', '513130.XSHG': '德国',
    '513080.XSHG': '法国', '513310.XSHG': '东南亚',
    '513730.XSHG': '半导体(韩国)', '159792.XSHE': '半导体',
    '159920.XSHE': '沪深300', '510300.XSHG': '沪深300',
    '510500.XSHG': '中证500', '510210.XSHG': '上证50',
    '159915.XSHE': '创业板', '588080.XSHG': '科创50',
    '512100.XSHG': '中证1000', '563360.XSHG': '中证1000', '563300.XSHG': '中证1000',
    '512890.XSHG': '红利', '159967.XSHE': '红利',
    '512040.XSHG': '金融',
}

# 回测区间
START_DATE = '2020-01-02'
END_DATE = '2026-06-30'

# 因子观察的未来收益窗口
N_DAYS_LIST = [5, 20, 60]


def _convert_code(jq_code):
    """聚宽代码 -> HData pivot 代码：518880.XSHG -> 518880.SH"""
    if jq_code.endswith('.XSHG'):
        return jq_code.replace('.XSHG', '.SH')
    if jq_code.endswith('.XSHE'):
        return jq_code.replace('.XSHE', '.SZ')
    return jq_code


def load_daily_prices():
    """从 HData 1d_etf_price zip 文件读取日线收盘价

    1d_etf_price/{year}.zip 内是按日 parquet（YYYYMMDD.parquet），
    每个含全市场 ETF 当日数据，列含 code/date/close/adj_factor 等。
    code 格式为 .SZ/.SH。
    我们提取 36 只 ETF 的前复权收盘价（close × adj_factor / 最新 adj_factor）。
    """
    HDATA_ROOT = Path(os.environ.get('HDATA_ROOT', r'D:\Work Space\HData'))
    etf_dir = HDATA_ROOT / 'data' / 'raw' / '指数与ETF数据' / '1d_etf_price'

    # ETF 代码映射：聚宽 -> HData
    code_map = {jq: _convert_code(jq) for jq in ALL_ETFS}
    hd_codes = set(code_map.values())

    # 按年读取
    all_dfs = []
    for yr in range(2020, 2027):
        zp = etf_dir / f'{yr}.zip'
        if not zp.exists():
            print(f"  [跳过] {zp.name} 不存在")
            continue
        import zipfile
        import io
        with zipfile.ZipFile(zp, 'r') as z:
            names = sorted(z.namelist())
            yr_dfs = []
            for nm in names:
                if not nm.endswith('.parquet'):
                    continue
                with z.open(nm) as f:
                    df = pd.read_parquet(io.BytesIO(f.read()))
                # 筛选我们的 ETF
                df = df[df['code'].isin(hd_codes)]
                if len(df) > 0:
                    yr_dfs.append(df[['code', 'date', 'close', 'adj_factor']])
        if yr_dfs:
            yr_df = pd.concat(yr_dfs, ignore_index=True)
            all_dfs.append(yr_df)
            print(f"  [加载] {yr}: {len(names)} 日, 筛选后 {len(yr_df)} 行")

    if not all_dfs:
        raise FileNotFoundError("未找到 1d_etf_price zip")

    raw = pd.concat(all_dfs, ignore_index=True)
    raw['date'] = pd.to_datetime(raw['date'], format='%Y%m%d')

    # 前复权：close × adj_factor / 该 ETF 最新 adj_factor
    # 注意：P3-02 是因子论证，不是回测对齐，前复权用全局最新 adj_factor 即可
    raw['adj_close'] = raw['close'] * raw['adj_factor']
    # 按 ETF 分组，除以最新 adj_factor
    latest_adj = raw.groupby('code')['adj_factor'].last()
    raw['fq_close'] = raw.apply(lambda r: r['close'] * r['adj_factor'] / latest_adj[r['code']], axis=1)

    # 透视成 date × code
    piv = raw.pivot(index='date', columns='code', values='fq_close').sort_index()

    # 列名改回聚宽代码
    rev_map = {v: k for k, v in code_map.items()}
    piv = piv.rename(columns=rev_map)

    print(f"[HDATA] 价格矩阵: {piv.shape[0]} 交易日 × {piv.shape[1]} ETF")
    return piv


def calc_momentum_score(prices, lookback=25):
    """计算动量得分 = 年化收益 × R²（与策略一致，加权线性回归）"""
    if len(prices) < lookback + 1:
        return np.nan
    recent = prices[-(lookback + 1):].values
    if np.any(np.isnan(recent)) or np.any(recent <= 0):
        return np.nan
    y = np.log(recent)
    x = np.arange(len(y))
    w = np.linspace(1, 2, len(y))
    slope, intercept = np.polyfit(x, y, 1, w=w)
    ann_ret = math.exp(slope * 250) - 1
    ss_res = np.sum(w * (y - (slope * x + intercept)) ** 2)
    ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    return ann_ret * r2


def build_portfolio_nav(price_matrix):
    """构建等权组合日频净值（独立 MTM，用日线收盘价）"""
    # 原池 8 只等权（含 511880 货币）
    default_prices = price_matrix[DEFAULT_POOL].dropna(how='all')
    # 大池独有 28 只等权
    bak_only_prices = price_matrix[BAK_ONLY].dropna(how='all')

    # 等权组合：每日对有价格的 ETF 等权
    def eq_weight_nav(prices_df, name):
        # 对每行，用有值的 ETF 等权计算当日收益
        rets = prices_df.pct_change()
        # 首日设为 0
        rets.iloc[0] = 0
        # 等权：每行非 nan 的列各占 1/n_valid
        n_valid = rets.notna().sum(axis=1)
        eq_ret = rets.sum(axis=1) / n_valid.replace(0, np.nan)
        eq_ret = eq_ret.fillna(0)
        nav = (1 + eq_ret).cumprod()
        nav.name = name
        return nav

    default_nav = eq_weight_nav(default_prices, 'default_pool')
    bak_nav = eq_weight_nav(bak_only_prices, 'bak_only')

    # 主题组合：按主题分组等权
    theme_navs = {}
    themes = sorted(set(ETF_THEME.values()))
    for theme in themes:
        etfs = [e for e in ALL_ETFS if ETF_THEME.get(e) == theme]
        if len(etfs) < 2:  # 单标的主题不建组合
            continue
        theme_prices = price_matrix[etfs].dropna(how='all')
        if len(theme_prices) < 30:
            continue
        theme_navs[theme] = eq_weight_nav(theme_prices, theme)

    # 合并
    all_nav = pd.DataFrame({'default_pool': default_nav, 'bak_only': bak_nav})
    for t, n in theme_navs.items():
        all_nav[f'theme_{t}'] = n

    return all_nav, theme_navs


def calc_factors(price_matrix):
    """计算 4 个交易前因子的 T-1 日频序列"""
    dates = price_matrix.index
    factors = []

    for i in range(60, len(dates)):  # 从第 60 日开始（保证 60 日动量有数据）
        t = dates[i]
        # T-1 数据：用截至 t-1 的价格
        pm_t1 = price_matrix.loc[:dates[i - 1]]

        # 原池 8 只的 25 日动量得分
        default_scores = []
        for etf in DEFAULT_POOL:
            if etf in pm_t1.columns:
                s = calc_momentum_score(pm_t1[etf].dropna(), 25)
                if not np.isnan(s):
                    default_scores.append(s)
        default_mean = np.mean(default_scores) if default_scores else np.nan

        # 大池独有 28 只的 25 日动量得分
        bak_scores = []
        for etf in BAK_ONLY:
            if etf in pm_t1.columns:
                s = calc_momentum_score(pm_t1[etf].dropna(), 25)
                if not np.isnan(s):
                    bak_scores.append(s)
        bak_mean = np.mean(bak_scores) if bak_scores else np.nan

        # 因子1：大池独有 - 原池 动量差
        factor1 = bak_mean - default_mean if not (np.isnan(bak_mean) or np.isnan(default_mean)) else np.nan

        # 因子2：TopN 集中度（37只按 25 日动量得分排序，Top2 中大池独有占比）
        all_scores = {}
        for etf in DEFAULT_POOL + BAK_ONLY:
            if etf in pm_t1.columns:
                s = calc_momentum_score(pm_t1[etf].dropna(), 25)
                if not np.isnan(s):
                    all_scores[etf] = s
        if len(all_scores) >= 2:
            top2 = sorted(all_scores.items(), key=lambda x: -x[1])[:2]
            bak_in_top2 = sum(1 for e, _ in top2 if e in BAK_ONLY)
            factor2 = bak_in_top2 / 2.0
        else:
            factor2 = np.nan

        # 因子4：大池独有 Top3 主题共振
        # 大池独有按主题分组，取 Top3 主题（主题内 ETF 平均得分），算 Top3 主题平均动量
        theme_scores = defaultdict(list)
        for etf, s in all_scores.items():
            if etf in BAK_ONLY:
                theme = ETF_THEME.get(etf, '其他')
                theme_scores[theme].append(s)
        theme_mean = {t: np.mean(ss) for t, ss in theme_scores.items()}
        if len(theme_mean) >= 3:
            top3_themes = sorted(theme_mean.items(), key=lambda x: -x[1])[:3]
            factor4 = np.mean([s for _, s in top3_themes])
        else:
            factor4 = np.nan

        # 因子5：原池 Top1/Top2 强度（原池 8 只得分最高的 2 只的平均）
        default_all = {e: s for e, s in all_scores.items() if e in DEFAULT_POOL}
        if len(default_all) >= 2:
            top2_default = sorted(default_all.items(), key=lambda x: -x[1])[:2]
            factor5 = np.mean([s for _, s in top2_default])
        else:
            factor5 = np.nan

        factors.append({
            'date': t.strftime('%Y-%m-%d'),
            'factor1_momentum_diff': factor1,
            'factor2_topn_bak_share': factor2,
            'factor4_top3_theme_momentum': factor4,
            'factor5_default_top2_strength': factor5,
            'bak_mean_score': bak_mean,
            'default_mean_score': default_mean,
        })

    return pd.DataFrame(factors)


def calc_future_returns(price_matrix, n_days_list):
    """计算原池/大池独有等权组合的未来 N 日收益（用于分桶统计）"""
    default_prices = price_matrix[DEFAULT_POOL].dropna(how='all')
    bak_prices = price_matrix[BAK_ONLY].dropna(how='all')

    def eq_weight_returns(prices_df):
        rets = prices_df.pct_change()
        rets.iloc[0] = 0
        n_valid = rets.notna().sum(axis=1)
        eq_ret = rets.sum(axis=1) / n_valid.replace(0, np.nan)
        return eq_ret.fillna(0)

    default_ret = eq_weight_returns(default_prices)
    bak_ret = eq_weight_returns(bak_prices)

    # 未来 N 日累计收益
    future = pd.DataFrame(index=price_matrix.index)
    for n in n_days_list:
        # 原池未来 N 日累计收益
        future[f'default_ret_{n}d'] = (1 + default_ret).rolling(n).apply(np.prod, raw=True) - 1
        # 大池独有未来 N 日累计收益
        future[f'bak_ret_{n}d'] = (1 + bak_ret).rolling(n).apply(np.prod, raw=True) - 1
        # 超额 = 大池独有 - 原池
        future[f'excess_ret_{n}d'] = future[f'bak_ret_{n}d'] - future[f'default_ret_{n}d']

    return future


def bucket_stats(factor_series, future_returns, factor_name, n_buckets=3):
    """因子分桶统计：高/中/低桶的未来收益均值 + 按年份拆开

    对连续因子用 qcut 分桶；对分类型因子（如因子2 取值 0/0.5/1）直接按值分桶。
    """
    # 合并因子和未来收益（统一 date 类型）
    fs = factor_series[['date', factor_name]].copy()
    fs['date'] = pd.to_datetime(fs['date'])
    df = fs.merge(future_returns, left_on='date', right_index=True, how='inner')
    df = df.dropna(subset=[factor_name])

    # 判断是否为分类型因子（唯一值 ≤ 5 个）
    unique_vals = sorted(df[factor_name].unique())
    is_categorical = len(unique_vals) <= 5

    results = {'full_sample': {}, 'by_year': {}, 'is_categorical': is_categorical,
               'unique_values': [float(v) for v in unique_vals]}

    for n in N_DAYS_LIST:
        ret_col = f'excess_ret_{n}d'
        sub = df.dropna(subset=[ret_col]).copy()
        if len(sub) < 30:
            continue

        # 全样本分桶
        if is_categorical:
            # 分类型：直接按值分组
            sub['bucket'] = sub[factor_name].astype(str)
            full = sub.groupby('bucket')[ret_col].agg(['mean', 'std', 'count']).to_dict('index')
            results['full_sample'][f'{n}d'] = {
                k: {
                    'mean': full.get(k, {}).get('mean', None),
                    'count': int(full.get(k, {}).get('count', 0)),
                } for k in [str(v) for v in unique_vals]
            }
        else:
            try:
                sub['bucket'] = pd.qcut(sub[factor_name], n_buckets, labels=['low', 'mid', 'high'])
            except Exception:
                continue
            full = sub.groupby('bucket')[ret_col].agg(['mean', 'std', 'count']).to_dict('index')
            results['full_sample'][f'{n}d'] = {
                'low_mean': full.get('low', {}).get('mean', None),
                'mid_mean': full.get('mid', {}).get('mean', None),
                'high_mean': full.get('high', {}).get('mean', None),
                'low_count': int(full.get('low', {}).get('count', 0)),
                'mid_count': int(full.get('mid', {}).get('count', 0)),
                'high_count': int(full.get('high', {}).get('count', 0)),
            }

        # 按年份拆开
        results['by_year'][f'{n}d'] = {}
        sub['year'] = sub['date'].dt.strftime('%Y')
        for yr in sorted(sub['year'].unique()):
            yr_sub = sub[sub['year'] == yr].copy()
            if len(yr_sub) < 10:
                continue
            if is_categorical:
                yr_stats = yr_sub.groupby('bucket')[ret_col].agg(['mean', 'count']).to_dict('index')
                results['by_year'][f'{n}d'][yr] = {
                    k: {
                        'mean': yr_stats.get(k, {}).get('mean', None),
                        'count': int(yr_stats.get(k, {}).get('count', 0)),
                    } for k in [str(v) for v in unique_vals]
                }
            else:
                try:
                    q_low = sub[factor_name].quantile(1 / n_buckets)
                    q_high = sub[factor_name].quantile(2 / n_buckets)
                    yr_sub['bucket'] = pd.cut(yr_sub[factor_name],
                                              bins=[-np.inf, q_low, q_high, np.inf],
                                              labels=['low', 'mid', 'high'])
                except Exception:
                    continue
                yr_stats = yr_sub.groupby('bucket')[ret_col].agg(['mean', 'count']).to_dict('index')
                results['by_year'][f'{n}d'][yr] = {
                    'low_mean': yr_stats.get('low', {}).get('mean', None),
                    'mid_mean': yr_stats.get('mid', {}).get('mean', None),
                    'high_mean': yr_stats.get('high', {}).get('mean', None),
                    'low_count': int(yr_stats.get('low', {}).get('count', 0)),
                    'mid_count': int(yr_stats.get('mid', {}).get('count', 0)),
                    'high_count': int(yr_stats.get('high', {}).get('count', 0)),
                }

    return results


def calc_posthoc_capital_usage():
    """事后诊断变量：大池独有资金占用率（从 P3-01 归因数据）"""
    results = {}
    for w in ['2020', '2021', '2022', '2023', '2024', '2025', '2026H1']:
        with open(ATTR_DIR / f'pool_decomposition_{w}.json', encoding='utf-8') as f:
            d = json.load(f)
        results[w] = {g: d[g] for g in ['A', 'B', 'C', 'D']}
    return results


def main():
    print(f"{'=' * 80}")
    print(f"P3-02 候选切换因子的因果论证")
    print(f"{'=' * 80}")

    # 1. 加载日线数据
    price_matrix = load_daily_prices()

    # 2. 构建组合净值（独立 MTM）
    print(f"\n[NAV] 构建等权组合日频净值...")
    all_nav, theme_navs = build_portfolio_nav(price_matrix)
    all_nav.index.name = 'date'
    all_nav.to_json(OUT_DIR / 'daily_portfolio_nav.json', orient='index', date_format='iso')
    print(f"[NAV] 净值矩阵: {all_nav.shape}, 主题数: {len(theme_navs)}")

    # 3. 计算因子 T-1 序列
    print(f"\n[FACTOR] 计算 4 个交易前因子 T-1 序列...")
    factor_df = calc_factors(price_matrix)
    factor_df.to_json(OUT_DIR / 'factor_series.json', orient='records', date_format='iso')
    print(f"[FACTOR] 因子序列: {len(factor_df)} 日")

    # 4. 计算未来收益
    print(f"\n[RET] 计算未来收益...")
    future_df = calc_future_returns(price_matrix, N_DAYS_LIST)

    # 5. 分桶统计
    print(f"\n[BUCKET] 因子分桶统计...")
    factor_names = {
        'factor1_momentum_diff': '因子1：大池独有-原池动量差',
        'factor2_topn_bak_share': '因子2：TopN大池独有占比',
        'factor4_top3_theme_momentum': '因子4：大池独有Top3主题动量',
        'factor5_default_top2_strength': '因子5：原池Top1/2强度',
    }
    all_bucket = {}
    for fn, label in factor_names.items():
        print(f"  - {label}")
        all_bucket[fn] = {
            'label': label,
            'stats': bucket_stats(factor_df, future_df, fn),
        }

    # 6. 事后诊断变量
    print(f"\n[POSTHOC] 事后诊断变量（资金占用率）...")
    posthoc = calc_posthoc_capital_usage()

    # 7. 保存
    with open(OUT_DIR / 'bucket_stats.json', 'w', encoding='utf-8') as f:
        json.dump(all_bucket, f, indent=2, ensure_ascii=False, default=str)
    with open(OUT_DIR / 'posthoc_capital_usage.json', 'w', encoding='utf-8') as f:
        json.dump(posthoc, f, indent=2, ensure_ascii=False, default=str)

    # 8. 打印摘要
    print(f"\n{'=' * 80}")
    print(f"分桶统计摘要（超额收益 = 大池独有 - 原池，未来 20 日）")
    print(f"{'=' * 80}")
    for fn, label in factor_names.items():
        s = all_bucket[fn]['stats']['full_sample'].get('20d', {})
        is_cat = all_bucket[fn]['stats'].get('is_categorical', False)
        print(f"\n{label}")
        if is_cat:
            uvs = all_bucket[fn]['stats'].get('unique_values', [])
            # 全样本
            parts = [f"{k}={s.get(k, {}).get('mean', 0):.4f}(n={s.get(k, {}).get('count', 0)})" for k in [str(v) for v in uvs]]
            print(f"  全样本: {'  '.join(parts)}")
            for yr in ['2020', '2021', '2022', '2023', '2024', '2025', '2026']:
                ys = all_bucket[fn]['stats']['by_year'].get('20d', {}).get(yr, {})
                if ys:
                    parts = [f"{k}={ys.get(k, {}).get('mean', 0):.4f}(n={ys.get(k, {}).get('count', 0)})" for k in [str(v) for v in uvs]]
                    print(f"  {yr}: {'  '.join(parts)}")
        else:
            print(f"  全样本: low={s.get('low_mean', 0):.4f}  mid={s.get('mid_mean', 0):.4f}  high={s.get('high_mean', 0):.4f}  "
                  f"(n={s.get('low_count',0)}/{s.get('mid_count',0)}/{s.get('high_count',0)})")
            for yr in ['2020', '2021', '2022', '2023', '2024', '2025', '2026']:
                ys = all_bucket[fn]['stats']['by_year'].get('20d', {}).get(yr, {})
                if ys:
                    print(f"  {yr}: low={ys.get('low_mean',0):.4f}  mid={ys.get('mid_mean',0):.4f}  high={ys.get('high_mean',0):.4f}  "
                          f"(n={ys.get('low_count',0)}/{ys.get('mid_count',0)}/{ys.get('high_count',0)})")

    print(f"\n输出目录: {OUT_DIR}")


if __name__ == '__main__':
    main()
