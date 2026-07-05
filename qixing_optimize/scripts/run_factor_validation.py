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

# ==== 池定义（与策略源码 g.etf_pool + run_sweep.py POOL_BAK 完全一致） ====
# 原风险池 7 只（策略源码 g.etf_pool，不含防御 ETF）
ORIGINAL_RISK_POOL = [
    '518880.XSHG',   # 黄金ETF
    '159985.XSHE',   # 豆粕ETF
    '501018.XSHG',   # 南方原油
    '161226.XSHE',   # 白银LOF
    '513100.XSHG',   # 纳指ETF
    '159915.XSHE',   # 创业板ETF
    '511220.XSHG',   # 城投债ETF
]
# 防御 ETF（货币基金，单独处理，不参与 TopN 计算）
DEFENSIVE_ETF = '511880.XSHG'
# 大池 37 只（run_sweep.py POOL_BAK，已剔除缺失的 159201.XSHE）
POOL_BAK_37 = [
    '518880.XSHG', '159980.XSHE', '159985.XSHE', '501018.XSHG', '161226.XSHE', '159981.XSHE',
    '513100.XSHG', '159509.XSHE', '513290.XSHG', '513500.XSHG', '159529.XSHE', '513400.XSHG',
    '513520.XSHG', '513030.XSHG', '513080.XSHG', '513310.XSHG', '513730.XSHG',
    '159792.XSHE', '513130.XSHG', '513050.XSHG', '159920.XSHE', '513690.XSHG',
    '510300.XSHG', '510500.XSHG', '510050.XSHG', '510210.XSHG', '159915.XSHE', '588080.XSHG',
    '512100.XSHG', '563360.XSHG', '563300.XSHG',
    '512890.XSHG', '159967.XSHE', '512040.XSHG',
    '511380.XSHG', '511010.XSHG', '511220.XSHG',
]
# 大池独有 30 只 = POOL_BAK_37 - ORIGINAL_RISK_POOL
BAK_ONLY = [e for e in POOL_BAK_37 if e not in ORIGINAL_RISK_POOL]
assert len(BAK_ONLY) == 30, f"大池独有应为 30 只，实际 {len(BAK_ONLY)}"
# 稳健性口径：原风险池 + 防御 ETF = 8 只
DEFAULT_WITH_DEFENSIVE = ORIGINAL_RISK_POOL + [DEFENSIVE_ETF]
# 主分析池合计：原风险池 7 + 大池独有 30 = 37 只风险 ETF
ALL_RISK_ETFS = ORIGINAL_RISK_POOL + BAK_ONLY  # 37 只
# 含防御稳健性口径：37 + 1 = 38 只
ALL_ETFS = ALL_RISK_ETFS + [DEFENSIVE_ETF]  # 38 只

# ==== 主题映射（按策略源码注释） ====
ETF_THEME = {
    '518880.XSHG': '黄金', '159980.XSHE': '有色金属', '159985.XSHE': '豆粕',
    '501018.XSHG': '原油', '159981.XSHE': '能源化工',
    '161226.XSHE': '白银',
    '513100.XSHG': '纳指', '159509.XSHE': '纳指科技', '513290.XSHG': '纳指生物',
    '513500.XSHG': '标普500', '159529.XSHE': '标普消费', '513400.XSHG': '道琼斯',
    '513520.XSHG': '日经', '513030.XSHG': '德国', '513080.XSHG': '法国',
    '513310.XSHG': '中韩半导体', '513730.XSHG': '东南亚',
    '159792.XSHE': '港股互联', '513130.XSHG': '恒生科技', '513050.XSHG': '中概互联',
    '159920.XSHE': '恒生', '513690.XSHG': '港股红利',
    '510300.XSHG': '沪深300', '510500.XSHG': '中证500', '510050.XSHG': '上证50',
    '510210.XSHG': '上证', '159915.XSHE': '创业板', '588080.XSHG': '科创50',
    '512100.XSHG': '中证1000', '563360.XSHG': 'A500', '563300.XSHG': '中证2000',
    '512890.XSHG': '红利低波', '159967.XSHE': '创业板成长', '512040.XSHG': '价值',
    '511380.XSHG': '可转债', '511010.XSHG': '国债', '511220.XSHG': '城投债',
    '511880.XSHG': '货币(防御)',
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
    我们提取 38 只 ETF 的前复权收盘价（close × adj_factor / 最新 adj_factor）。
    含原风险池 7 + 大池独有 30 + 防御 ETF 1 = 38 只。
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
    """构建等权组合日频净值（独立 MTM，用日线收盘价）

    主口径：原风险池 7 只 vs 大池独有 30 只
    稳健性口径：原风险池 7 + 防御 1 = 8 只 vs 大池独有 30 只
    """
    # 主口径：原风险池 7 只
    risk_pool_prices = price_matrix[ORIGINAL_RISK_POOL].dropna(how='all')
    # 稳健性口径：原风险池 + 防御 ETF = 8 只
    default_w_def_prices = price_matrix[DEFAULT_WITH_DEFENSIVE].dropna(how='all')
    # 大池独有 30 只
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

    risk_pool_nav = eq_weight_nav(risk_pool_prices, 'risk_pool_7')
    default_w_def_nav = eq_weight_nav(default_w_def_prices, 'default_w_def_8')
    bak_nav = eq_weight_nav(bak_only_prices, 'bak_only_30')

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
    all_nav = pd.DataFrame({
        'risk_pool_7': risk_pool_nav,           # 主口径：原风险池 7 只
        'default_w_def_8': default_w_def_nav,   # 稳健性口径：原风险池+防御 8 只
        'bak_only_30': bak_nav,                 # 大池独有 30 只
    })
    for t, n in theme_navs.items():
        all_nav[f'theme_{t}'] = n

    return all_nav, theme_navs


def calc_factors(price_matrix):
    """计算 4 个交易前因子的 T-1 日频序列

    主口径：原风险池 7 只 vs 大池独有 30 只
    稳健性口径：原风险池+防御 8 只 vs 大池独有 30 只
    防御 ETF 不参与 TopN 计算（因子2/4）
    """
    dates = price_matrix.index
    factors = []

    for i in range(60, len(dates)):  # 从第 60 日开始（保证 60 日动量有数据）
        t = dates[i]
        # T-1 数据：用截至 t-1 的价格
        pm_t1 = price_matrix.loc[:dates[i - 1]]

        # 原风险池 7 只的 25 日动量得分
        risk_scores = []
        for etf in ORIGINAL_RISK_POOL:
            if etf in pm_t1.columns:
                s = calc_momentum_score(pm_t1[etf].dropna(), 25)
                if not np.isnan(s):
                    risk_scores.append(s)
        risk_mean = np.mean(risk_scores) if risk_scores else np.nan

        # 稳健性口径：原风险池 + 防御 ETF = 8 只的 25 日动量得分
        default_w_def_scores = []
        for etf in DEFAULT_WITH_DEFENSIVE:
            if etf in pm_t1.columns:
                s = calc_momentum_score(pm_t1[etf].dropna(), 25)
                if not np.isnan(s):
                    default_w_def_scores.append(s)
        default_w_def_mean = np.mean(default_w_def_scores) if default_w_def_scores else np.nan

        # 大池独有 30 只的 25 日动量得分
        bak_scores = []
        for etf in BAK_ONLY:
            if etf in pm_t1.columns:
                s = calc_momentum_score(pm_t1[etf].dropna(), 25)
                if not np.isnan(s):
                    bak_scores.append(s)
        bak_mean = np.mean(bak_scores) if bak_scores else np.nan

        # 因子1：大池独有 - 原风险池 动量差（主口径）
        factor1_main = bak_mean - risk_mean if not (np.isnan(bak_mean) or np.isnan(risk_mean)) else np.nan
        # 因子1 稳健性口径：大池独有 - (原风险池+防御)
        factor1_robust = bak_mean - default_w_def_mean if not (np.isnan(bak_mean) or np.isnan(default_w_def_mean)) else np.nan

        # 因子2：TopN 集中度（37 只风险 ETF 按动量排序，Top2 中大池独有占比）
        # 防御 ETF 不参与 TopN 计算
        all_scores = {}
        for etf in ALL_RISK_ETFS:  # 37 只风险 ETF，不含防御
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

        # 因子5：原风险池 Top1/Top2 强度（原风险池 7 只得分最高的 2 只的平均）
        # 主口径：仅原风险池 7 只
        risk_all = {e: s for e, s in all_scores.items() if e in ORIGINAL_RISK_POOL}
        if len(risk_all) >= 2:
            top2_risk = sorted(risk_all.items(), key=lambda x: -x[1])[:2]
            factor5_main = np.mean([s for _, s in top2_risk])
        else:
            factor5_main = np.nan
        # 稳健性口径：原风险池 + 防御 ETF = 8 只
        # 防御 ETF 单独算分
        def_score = np.nan
        if DEFENSIVE_ETF in pm_t1.columns:
            def_s = calc_momentum_score(pm_t1[DEFENSIVE_ETF].dropna(), 25)
            if not np.isnan(def_s):
                def_score = def_s
        risk_w_def_all = dict(risk_all)
        if not np.isnan(def_score):
            risk_w_def_all[DEFENSIVE_ETF] = def_score
        if len(risk_w_def_all) >= 2:
            top2_rwd = sorted(risk_w_def_all.items(), key=lambda x: -x[1])[:2]
            factor5_robust = np.mean([s for _, s in top2_rwd])
        else:
            factor5_robust = np.nan

        factors.append({
            'date': t.strftime('%Y-%m-%d'),
            'factor1_momentum_diff': factor1_main,            # 主口径
            'factor1_momentum_diff_robust': factor1_robust,   # 稳健性口径
            'factor2_topn_bak_share': factor2,
            'factor4_top3_theme_momentum': factor4,
            'factor5_default_top2_strength': factor5_main,    # 主口径
            'factor5_default_top2_strength_robust': factor5_robust,  # 稳健性口径
            'bak_mean_score': bak_mean,
            'risk_pool_mean_score': risk_mean,
            'default_w_def_mean_score': default_w_def_mean,
        })

    return pd.DataFrame(factors)


def calc_future_returns(price_matrix, n_days_list):
    """计算原风险池/大池独有等权组合的未来 N 日收益（用于分桶统计）

    未来 N 日收益口径：因子日期为 T（用 T-1 及以前数据），未来收益 = T 到 T+N-1 的累计收益。
    实现方法：rolling(n) 在索引 i 处算的是 [i-n+1, i] 的累计收益，
    用 shift(-(n-1)) 把索引 i+n-1 处的值移到索引 i，即得到 [i, i+n-1] 的未来 N 日收益。

    主口径超额 = 大池独有 - 原风险池7
    稳健性口径超额 = 大池独有 - (原风险池7+防御1)
    """
    risk_prices = price_matrix[ORIGINAL_RISK_POOL].dropna(how='all')
    default_w_def_prices = price_matrix[DEFAULT_WITH_DEFENSIVE].dropna(how='all')
    bak_prices = price_matrix[BAK_ONLY].dropna(how='all')

    def eq_weight_returns(prices_df):
        rets = prices_df.pct_change()
        rets.iloc[0] = 0
        n_valid = rets.notna().sum(axis=1)
        eq_ret = rets.sum(axis=1) / n_valid.replace(0, np.nan)
        return eq_ret.fillna(0)

    risk_ret = eq_weight_returns(risk_prices)
    default_w_def_ret = eq_weight_returns(default_w_def_prices)
    bak_ret = eq_weight_returns(bak_prices)

    # 未来 N 日累计收益（T 到 T+N-1）
    future = pd.DataFrame(index=price_matrix.index)
    for n in n_days_list:
        # rolling(n) 在 i 处算 [i-n+1, i] 累计收益；shift(-(n-1)) 后在 i 处得到 [i, i+n-1] 未来收益
        roll_risk = (1 + risk_ret).rolling(n).apply(np.prod, raw=True) - 1
        future[f'risk_ret_{n}d'] = roll_risk.shift(-(n - 1))
        roll_default_w_def = (1 + default_w_def_ret).rolling(n).apply(np.prod, raw=True) - 1
        future[f'default_w_def_ret_{n}d'] = roll_default_w_def.shift(-(n - 1))
        roll_bak = (1 + bak_ret).rolling(n).apply(np.prod, raw=True) - 1
        future[f'bak_ret_{n}d'] = roll_bak.shift(-(n - 1))
        # 主口径超额 = 大池独有 - 原风险池
        future[f'excess_ret_{n}d'] = future[f'bak_ret_{n}d'] - future[f'risk_ret_{n}d']
        # 稳健性口径超额 = 大池独有 - (原风险池+防御)
        future[f'excess_ret_robust_{n}d'] = future[f'bak_ret_{n}d'] - future[f'default_w_def_ret_{n}d']

    return future


def bucket_stats(factor_series, future_returns, factor_name, n_buckets=3, ret_prefix='excess_ret'):
    """因子分桶统计：高/中/低桶的未来收益均值 + 按年份拆开

    对连续因子用 qcut 分桶；对分类型因子（如因子2 取值 0/0.5/1）直接按值分桶。
    ret_prefix: 'excess_ret'（主口径）或 'excess_ret_robust'（稳健性口径）
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
               'unique_values': [float(v) for v in unique_vals], 'ret_prefix': ret_prefix}

    for n in N_DAYS_LIST:
        ret_col = f'{ret_prefix}_{n}d'
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

    # 5. 分桶统计（主口径 + 稳健性口径）
    print(f"\n[BUCKET] 因子分桶统计...")
    # 主口径因子（excess_ret = 大池独有 - 原风险池7）
    factor_names_main = {
        'factor1_momentum_diff': '因子1主口径：大池独有-原风险池7动量差',
        'factor2_topn_bak_share': '因子2：TopN大池独有占比（37风险ETF，防御不参与）',
        'factor4_top3_theme_momentum': '因子4：大池独有Top3主题动量',
        'factor5_default_top2_strength': '因子5主口径：原风险池7 Top1/2强度',
    }
    # 稳健性口径因子（excess_ret_robust = 大池独有 - (原风险池7+防御1)）
    factor_names_robust = {
        'factor1_momentum_diff_robust': '因子1稳健性：大池独有-(原风险池7+防御1)动量差',
        'factor5_default_top2_strength_robust': '因子5稳健性：(原风险池7+防御1) Top1/2强度',
    }
    all_bucket = {}
    print("  [主口径]")
    for fn, label in factor_names_main.items():
        print(f"    - {label}")
        # 因子2/4 用主口径 excess_ret；因子1/5 主口径也用 excess_ret
        all_bucket[fn] = {
            'label': label,
            'caliber': 'main',
            'stats': bucket_stats(factor_df, future_df, fn, ret_prefix='excess_ret'),
        }
    print("  [稳健性口径]")
    for fn, label in factor_names_robust.items():
        print(f"    - {label}")
        all_bucket[fn] = {
            'label': label,
            'caliber': 'robust',
            'stats': bucket_stats(factor_df, future_df, fn, ret_prefix='excess_ret_robust'),
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
    print(f"分桶统计摘要（未来 20 日超额收益）")
    print(f"主口径: 超额 = 大池独有 - 原风险池7 | 稳健性: 超额 = 大池独有 - (原风险池7+防御1)")
    print(f"{'=' * 80}")
    all_factor_names = {**factor_names_main, **factor_names_robust}
    for fn, label in all_factor_names.items():
        s = all_bucket[fn]['stats']['full_sample'].get('20d', {})
        is_cat = all_bucket[fn]['stats'].get('is_categorical', False)
        print(f"\n{label}")
        if is_cat:
            uvs = all_bucket[fn]['stats'].get('unique_values', [])
            def _fmt_cat(d, k):
                m = d.get(k, {}).get('mean')
                c = d.get(k, {}).get('count', 0)
                return f"{k}={m:.4f}(n={c})" if m is not None else f"{k}=N/A(n={c})"
            parts = [_fmt_cat(s, str(v)) for v in uvs]
            print(f"  全样本: {'  '.join(parts)}")
            for yr in ['2020', '2021', '2022', '2023', '2024', '2025', '2026']:
                ys = all_bucket[fn]['stats']['by_year'].get('20d', {}).get(yr, {})
                if ys:
                    parts = [_fmt_cat(ys, str(v)) for v in uvs]
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
