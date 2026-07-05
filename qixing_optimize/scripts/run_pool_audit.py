"""
P3-03 核心池/卫星池定义审计 + 非动量候选变量挖掘
=========================================================
基准提交：6d5d47c（P3-02 未来收益修正版）

子任务：
  A. ETF 逐只画像（37 风险 + 1 防御）
  B. 核心池定义审计（保留/争议/卫星 三类候选）
  C. 卫星池主题结构审计（2024 赢 / 2026H1 输 来源）
  D. 非动量候选变量挖掘（6 类变量，三层审计）

严格边界（用户硬约束）：
- 不生成切换规则，不给买卖阈值
- 不回测任何新策略，不优化参数
- 不用年度胜负反推规则
- 不把候选变量包装成"有效因子"
- 不宣布任何变量可实盘
- 不改 P1/P2/P3-01/P3-02 已冻结结论
- 不修改交易逻辑，不进入 P4
- 描述性分桶是"候选观察"，不是切换规则

池口径（P3-02 修正版冻结）：
- 原风险池 7 只（g.etf_pool，含 159915，不含 511880）
- 防御 ETF 511880 单独
- 大池 37 只（POOL_BAK_37，已剔 159201）
- 大池独有 30 只 = 37 - 7
"""
import json
import sys
import os
import zipfile
import io
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# ==== 路径 ====
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'qixing_optimize' / 'runs' / 'p3_03_pool_audit'
OUT_DIR.mkdir(parents=True, exist_ok=True)
ATTR_DIR = ROOT / 'qixing_optimize' / 'runs' / 'attribution'

# ==== HData 环境 ====
HDATA_ROOT = Path(os.environ.get('HDATA_ROOT', r'D:\Work Space\HData'))

# ==== 池定义（P3-02 修正版冻结，与策略源码 + run_sweep.py 完全一致） ====
ORIGINAL_RISK_POOL = [
    '518880.XSHG',   # 黄金ETF
    '159985.XSHE',   # 豆粕ETF
    '501018.XSHG',   # 南方原油
    '161226.XSHE',   # 白银LOF
    '513100.XSHG',   # 纳指ETF
    '159915.XSHE',   # 创业板ETF
    '511220.XSHG',   # 城投债ETF
]
DEFENSIVE_ETF = '511880.XSHG'
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
BAK_ONLY = [e for e in POOL_BAK_37 if e not in ORIGINAL_RISK_POOL]
assert len(BAK_ONLY) == 30
ALL_RISK_ETFS = ORIGINAL_RISK_POOL + BAK_ONLY  # 37 只
ALL_ETFS = ALL_RISK_ETFS + [DEFENSIVE_ETF]  # 38 只

# ==== 主题映射（细粒度，按策略源码注释） ====
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

# ==== 主题大类分组（用于子任务 C 卫星主题结构审计） ====
THEME_CATEGORY = {
    '黄金': '商品/贵金属', '有色金属': '商品/贵金属', '豆粕': '商品/贵金属',
    '原油': '能源', '能源化工': '能源', '白银': '商品/贵金属',
    '纳指': '海外科技', '纳指科技': '海外科技', '纳指生物': '海外科技',
    '标普500': '海外宽基', '标普消费': '海外宽基', '道琼斯': '海外宽基',
    '日经': '欧日东南亚', '德国': '欧日东南亚', '法国': '欧日东南亚',
    '东南亚': '欧日东南亚', '中韩半导体': '海外科技',
    '港股互联': '中概/港股', '恒生科技': '中概/港股', '中概互联': '中概/港股',
    '恒生': '中概/港股', '港股红利': '中概/港股',
    '沪深300': 'A股宽基', '中证500': 'A股宽基', '上证50': 'A股宽基',
    '上证': 'A股宽基', '创业板': 'A股成长', '科创50': 'A股成长',
    '中证1000': 'A股宽基', 'A500': 'A股宽基', '中证2000': 'A股宽基',
    '红利低波': '红利/价值', '创业板成长': 'A股成长', '价值': '红利/价值',
    '可转债': '债券/固收', '国债': '债券/固收', '城投债': '债券/固收',
    '货币(防御)': '防御',
}

# 回测区间
START_DATE = '2020-01-02'
END_DATE = '2026-06-30'
N_DAYS_LIST = [5, 20, 60]

# 年份窗口定义（与 P2/P3-01 一致）
YEAR_WINDOWS = {
    '2020': ('2020-01-02', '2020-12-31'),
    '2021': ('2021-01-04', '2021-12-31'),
    '2022': ('2022-01-04', '2022-12-30'),
    '2023': ('2023-01-03', '2023-12-29'),
    '2024': ('2024-01-02', '2024-12-31'),
    '2025': ('2025-01-02', '2025-12-31'),
    '2026H1': ('2026-01-02', '2026-06-30'),
}


def _convert_code(jq_code):
    if jq_code.endswith('.XSHG'):
        return jq_code.replace('.XSHG', '.SH')
    if jq_code.endswith('.XSHE'):
        return jq_code.replace('.XSHE', '.SZ')
    return jq_code


def load_daily_data():
    """从 HData 1d_etf_price 读取日线数据（含 close/high/low/vol/amount/adj_factor）

    返回 dict：
      price_close: date × code 前复权收盘价
      price_high: date × code 前复权最高价
      price_low: date × code 前复权最低价
      amount: date × code 成交额（元）
      vol: date × code 成交量（手）
    """
    etf_dir = HDATA_ROOT / 'data' / 'raw' / '指数与ETF数据' / '1d_etf_price'
    code_map = {jq: _convert_code(jq) for jq in ALL_ETFS}
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

    # 前复权
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
    print(f"[HDATA] amount 字段可用: {result['amount'].notna().sum().sum() > 0}")
    return result


# ============================================================
# 子任务 A：ETF 逐只画像
# ============================================================
def build_etf_profile(data):
    """对 38 只 ETF 逐只画像"""
    close = data['close']
    amount = data['amount']
    all_dates = close.index

    profiles = []
    for etf in ALL_ETFS:
        if etf not in close.columns:
            continue
        s = close[etf].dropna()
        amt = amount[etf].dropna() if etf in amount.columns else pd.Series(dtype=float)

        if len(s) < 30:
            profiles.append({
                'code': etf, 'theme': ETF_THEME.get(etf, '其他'),
                'theme_category': THEME_CATEGORY.get(ETF_THEME.get(etf, ''), '其他'),
                'is_original_risk': etf in ORIGINAL_RISK_POOL,
                'is_defensive': etf == DEFENSIVE_ETF,
                'is_bak_only': etf in BAK_ONLY,
                'data_start': str(s.index.min().date()) if len(s) > 0 else None,
                'data_end': str(s.index.max().date()) if len(s) > 0 else None,
                'n_valid_days': int(len(s)),
                'data_coverage': 0.0,
                'note': '数据不足',
            })
            continue

        # 日收益率
        rets = s.pct_change().dropna()
        # 数据覆盖率
        first_date = pd.Timestamp(START_DATE)
        last_date = pd.Timestamp(END_DATE)
        expected_days = len(all_dates[(all_dates >= first_date) & (all_dates <= last_date)])
        coverage = len(s) / expected_days if expected_days > 0 else 0.0

        # 长时间缺失检测：连续缺失 > 20 交易日
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq='B')
        s_reidx = s.reindex(full_idx)
        is_na = s_reidx.isna()
        # 找连续缺失段
        gap_lens = []
        cur = 0
        for v in is_na:
            if v:
                cur += 1
            else:
                if cur > 0:
                    gap_lens.append(cur)
                cur = 0
        if cur > 0:
            gap_lens.append(cur)
        max_gap = max(gap_lens) if gap_lens else 0
        long_gaps = [g for g in gap_lens if g > 20]

        # 波动率（年化）
        vol_annual = float(rets.std() * np.sqrt(252))

        # 最大回撤（全样本）
        nav = (1 + rets).cumprod()
        peak = nav.cummax()
        dd = (nav - peak) / peak
        max_dd = float(dd.min())

        # 成交额统计
        amt_mean = float(amt.mean()) if len(amt) > 0 else None
        amt_median = float(amt.median()) if len(amt) > 0 else None

        # 年度收益与回撤
        yearly = {}
        for yr, (ys, ye) in YEAR_WINDOWS.items():
            yr_s = s[(s.index >= pd.Timestamp(ys)) & (s.index <= pd.Timestamp(ye))]
            if len(yr_s) < 5:
                yearly[yr] = {'return_pct': None, 'max_dd_pct': None, 'n_days': int(len(yr_s))}
                continue
            yr_ret = float(yr_s.iloc[-1] / yr_s.iloc[0] - 1)
            yr_nav = (1 + yr_s.pct_change().dropna()).cumprod()
            yr_peak = yr_nav.cummax()
            yr_dd = float(((yr_nav - yr_peak) / yr_peak).min())
            yearly[yr] = {
                'return_pct': round(yr_ret * 100, 2),
                'max_dd_pct': round(yr_dd * 100, 2),
                'n_days': int(len(yr_s)),
            }

        # 单一年份贡献检测：某年收益占总收益绝对值超过 80%
        yr_rets = [v['return_pct'] for v in yearly.values() if v['return_pct'] is not None]
        single_year_dominant = False
        if len(yr_rets) >= 3:
            total_abs = sum(abs(r) for r in yr_rets)
            if total_abs > 0 and max(abs(r) for r in yr_rets) / total_abs > 0.8:
                single_year_dominant = True

        profiles.append({
            'code': etf,
            'theme': ETF_THEME.get(etf, '其他'),
            'theme_category': THEME_CATEGORY.get(ETF_THEME.get(etf, ''), '其他'),
            'is_original_risk': etf in ORIGINAL_RISK_POOL,
            'is_defensive': etf == DEFENSIVE_ETF,
            'is_bak_only': etf in BAK_ONLY,
            'data_start': str(s.index.min().date()),
            'data_end': str(s.index.max().date()),
            'n_valid_days': int(len(s)),
            'data_coverage': round(coverage * 100, 2),
            'amt_mean_yuan': round(amt_mean, 0) if amt_mean is not None else None,
            'amt_median_yuan': round(amt_median, 0) if amt_median is not None else None,
            'vol_annual_pct': round(vol_annual * 100, 2),
            'max_dd_pct': round(max_dd * 100, 2),
            'yearly': yearly,
            'max_gap_days': int(max_gap),
            'has_long_gap': len(long_gaps) > 0,
            'single_year_dominant': single_year_dominant,
            'note': '',
        })

    return profiles


# ============================================================
# 子任务 B：核心池定义审计
# ============================================================
def audit_core_pool(profiles):
    """基于事实审计，将 ETF 分为 核心池保留候选 / 核心池争议候选 / 卫星池候选

    判断维度（事实层，非规则）：
      - 跨周期数据完整（coverage >= 95%）
      - 流动性稳定（amt_median 不为 None 且 > 1e6）
      - 波动不过高（vol_annual < 50%）
      - 最大回撤相对可控（max_dd > -60%）
      - 不是单一年份贡献
      - 数据无长时间缺失
    """
    results = {'core_retain': [], 'core_disputed': [], 'satellite': []}

    for p in profiles:
        if p['is_defensive']:
            p['pool_class'] = '防御(单独)'
            p['audit_reason'] = '防御 ETF，单独处理，不参与核心/卫星分类'
            results['satellite'].append(p)
            continue

        # 事实判断
        coverage_ok = p['data_coverage'] >= 95.0
        liquidity_ok = p['amt_median_yuan'] is not None and p['amt_median_yuan'] > 1e6
        vol_ok = p['vol_annual_pct'] < 50.0
        dd_ok = p['max_dd_pct'] > -60.0
        no_long_gap = not p['has_long_gap']
        no_single_year = not p['single_year_dominant']

        passed = sum([coverage_ok, liquidity_ok, vol_ok, dd_ok, no_long_gap, no_single_year])
        reasons = []
        if not coverage_ok:
            reasons.append(f"数据覆盖率{p['data_coverage']}%<95%")
        if not liquidity_ok:
            reasons.append("流动性不足(成交额中位数<1e6或缺失)")
        if not vol_ok:
            reasons.append(f"年化波动{p['vol_annual_pct']}%>=50%")
        if not dd_ok:
            reasons.append(f"最大回撤{p['max_dd_pct']}%<=-60%")
        if not no_long_gap:
            reasons.append(f"存在长缺失(最大gap={p['max_gap_days']}天)")
        if not no_single_year:
            reasons.append("单一年份贡献主导")

        p['audit_passed_count'] = passed
        p['audit_reason'] = '; '.join(reasons) if reasons else '全部维度通过'

        if passed >= 6:
            p['pool_class'] = '核心池保留候选'
            results['core_retain'].append(p)
        elif passed >= 4:
            p['pool_class'] = '核心池争议候选'
            results['core_disputed'].append(p)
        else:
            p['pool_class'] = '卫星池候选'
            results['satellite'].append(p)

    return results


# ============================================================
# 子任务 C：卫星池主题结构审计
# ============================================================
def build_theme_profile(data, profiles):
    """按主题大类分组，输出主题级统计"""
    close = data['close']
    amount = data['amount']

    # 按主题大类分组
    cat_etfs = defaultdict(list)
    for etf in ALL_RISK_ETFS:
        theme = ETF_THEME.get(etf, '其他')
        cat = THEME_CATEGORY.get(theme, '其他')
        cat_etfs[cat].append(etf)

    theme_profiles = []
    for cat, etfs in cat_etfs.items():
        cat_prices = close[etfs].dropna(how='all')
        if len(cat_prices) < 30:
            continue

        # 等权组合净值
        rets = cat_prices.pct_change()
        rets.iloc[0] = 0
        n_valid = rets.notna().sum(axis=1)
        eq_ret = rets.sum(axis=1) / n_valid.replace(0, np.nan)
        eq_ret = eq_ret.fillna(0)
        nav = (1 + eq_ret).cumprod()

        # 波动率
        vol_annual = float(eq_ret.std() * np.sqrt(252))
        # 最大回撤
        peak = nav.cummax()
        dd = (nav - peak) / peak
        max_dd = float(dd.min())

        # 年度收益
        yearly = {}
        for yr, (ys, ye) in YEAR_WINDOWS.items():
            yr_nav = nav[(nav.index >= pd.Timestamp(ys)) & (nav.index <= pd.Timestamp(ye))]
            if len(yr_nav) < 5:
                yearly[yr] = {'return_pct': None, 'n_days': int(len(yr_nav))}
                continue
            yr_ret = float(yr_nav.iloc[-1] / yr_nav.iloc[0] - 1)
            yearly[yr] = {'return_pct': round(yr_ret * 100, 2), 'n_days': int(len(yr_nav))}

        # 单一年份依赖检测
        yr_rets = [v['return_pct'] for v in yearly.values() if v['return_pct'] is not None]
        single_year_dominant = False
        dominant_year = None
        if len(yr_rets) >= 3:
            total_abs = sum(abs(r) for r in yr_rets)
            if total_abs > 0:
                max_abs = max(abs(r) for r in yr_rets)
                if max_abs / total_abs > 0.7:
                    single_year_dominant = True
                    # 找出主导年份
                    for yr, v in yearly.items():
                        if v['return_pct'] is not None and abs(v['return_pct']) == max_abs:
                            dominant_year = yr
                            break

        # 成交额
        cat_amt = amount[etfs].dropna(how='all')
        amt_mean = float(cat_amt.mean().mean()) if len(cat_amt) > 0 else None

        theme_profiles.append({
            'theme_category': cat,
            'etfs': etfs,
            'n_etfs': len(etfs),
            'data_start': str(cat_prices.index.min().date()),
            'n_days': int(len(cat_prices)),
            'vol_annual_pct': round(vol_annual * 100, 2),
            'max_dd_pct': round(max_dd * 100, 2),
            'amt_mean_yuan': round(amt_mean, 0) if amt_mean is not None else None,
            'yearly': yearly,
            'single_year_dominant': single_year_dominant,
            'dominant_year': dominant_year,
            'is_bak_only_dominant': sum(1 for e in etfs if e in BAK_ONLY) >= len(etfs) / 2,
        })

    return theme_profiles


def analyze_2024_2026_source(data, profiles):
    """重点解释：2024 大池赢 vs 2026H1 大池输 来自哪些主题

    用 HData 独立 MTM 计算各主题大类在 2024 和 2026H1 的等权收益
    """
    close = data['close']

    cat_etfs = defaultdict(list)
    for etf in ALL_RISK_ETFS:
        theme = ETF_THEME.get(etf, '其他')
        cat = THEME_CATEGORY.get(theme, '其他')
        cat_etfs[cat].append(etf)

    results = {'2024': {}, '2026H1': {}}
    for yr in ['2024', '2026H1']:
        ys, ye = YEAR_WINDOWS[yr]
        mask = (close.index >= pd.Timestamp(ys)) & (close.index <= pd.Timestamp(ye))
        for cat, etfs in cat_etfs.items():
            cat_prices = close.loc[mask, etfs].dropna(how='all')
            if len(cat_prices) < 5:
                results[yr][cat] = None
                continue
            rets = cat_prices.pct_change()
            rets.iloc[0] = 0
            n_valid = rets.notna().sum(axis=1)
            eq_ret = rets.sum(axis=1) / n_valid.replace(0, np.nan)
            eq_ret = eq_ret.fillna(0)
            yr_ret = float((1 + eq_ret).prod() - 1)
            results[yr][cat] = round(yr_ret * 100, 2)

    return results


# ============================================================
# 子任务 D：非动量候选变量挖掘（6 类）
# ============================================================
def eq_weight_return(prices_df):
    """等权组合日收益"""
    rets = prices_df.pct_change()
    rets.iloc[0] = 0
    n_valid = rets.notna().sum(axis=1)
    eq_ret = rets.sum(axis=1) / n_valid.replace(0, np.nan)
    return eq_ret.fillna(0)


def calc_future_excess_return(data, n_days):
    """计算未来 N 日超额收益（大池独有 - 原风险池），用于分桶观察

    修正版：用 shift(-(n-1)) 得到 T 到 T+N-1 真正未来收益
    """
    close = data['close']
    risk_ret = eq_weight_return(close[ORIGINAL_RISK_POOL])
    bak_ret = eq_weight_return(close[BAK_ONLY])
    roll_bak = (1 + bak_ret).rolling(n_days).apply(np.prod, raw=True) - 1
    roll_risk = (1 + risk_ret).rolling(n_days).apply(np.prod, raw=True) - 1
    future_bak = roll_bak.shift(-(n_days - 1))
    future_risk = roll_risk.shift(-(n_days - 1))
    return future_bak - future_risk


def calc_candidate_variables(data):
    """计算 6 类非动量候选变量的 T-1 日频序列

    所有变量用 T-1 数据，避免未来函数。
    """
    close = data['close']
    high = data['high']
    amount = data['amount']

    # 原风险池/大池独有等权日收益
    risk_ret = eq_weight_return(close[ORIGINAL_RISK_POOL])
    bak_ret = eq_weight_return(close[BAK_ONLY])

    # 37 只风险 ETF 日收益矩阵
    all_ret = close[ALL_RISK_ETFS].pct_change()

    variables = {}
    var_meta = {}  # 变量元信息（可计算性审计）

    # ===== 1. 波动率变量 =====
    # 1.1 大池独有 ETF 波动率（20日滚动std，均值）
    bak_vol = all_ret[BAK_ONLY].rolling(20).std() * np.sqrt(252)
    variables['v1_bak_vol'] = bak_vol.mean(axis=1).shift(1)  # T-1
    var_meta['v1_bak_vol'] = {'category': '波动率', 'name': '大池独有ETF波动率(20日)',
                              'logic': '高波动环境下大池独有更易均值回归？',
                              'computable': True, 'future_function': False}

    # 1.2 原风险池波动率
    risk_vol = all_ret[ORIGINAL_RISK_POOL].rolling(20).std() * np.sqrt(252)
    variables['v1_risk_vol'] = risk_vol.mean(axis=1).shift(1)
    var_meta['v1_risk_vol'] = {'category': '波动率', 'name': '原风险池波动率(20日)',
                               'logic': '原池高波动时大池独有是否更值得开放？',
                               'computable': True, 'future_function': False}

    # 1.3 大池独有/原风险池 波动率比
    variables['v1_vol_ratio'] = (variables['v1_bak_vol'] / variables['v1_risk_vol'].replace(0, np.nan)).shift(0)
    var_meta['v1_vol_ratio'] = {'category': '波动率', 'name': '大池独有/原风险池波动率比',
                                'logic': '大池独有相对波动率越高，是否更易反转？',
                                'computable': True, 'future_function': False}

    # 1.4 波动率扩张（当前20日vol / 过去60日vol - 1）
    bak_vol_60 = all_ret[BAK_ONLY].rolling(60).std() * np.sqrt(252)
    vol_expansion = (bak_vol.mean(axis=1) / bak_vol_60.mean(axis=1).replace(0, np.nan) - 1)
    variables['v1_vol_expansion'] = vol_expansion.shift(1)
    var_meta['v1_vol_expansion'] = {'category': '波动率', 'name': '大池独有波动率扩张(20d/60d-1)',
                                    'logic': '波动率扩张时主题是否更易反转？',
                                    'computable': True, 'future_function': False}

    # ===== 2. 回撤位置变量 =====
    # 大池独有组合距 20/60/120 日高点的回撤
    bak_nav = (1 + bak_ret).cumprod()
    for window in [20, 60, 120]:
        rolling_max = bak_nav.rolling(window).max()
        dd = (bak_nav - rolling_max) / rolling_max
        variables[f'v2_bak_dd_{window}d'] = dd.shift(1)
        var_meta[f'v2_bak_dd_{window}d'] = {'category': '回撤位置',
                                            'name': f'大池独有距{window}日高点回撤',
                                            'logic': f'大池独有远离高点({window}d)时更易反转还是修复？',
                                            'computable': True, 'future_function': False}

    # 原风险池距 60 日高点回撤
    risk_nav = (1 + risk_ret).cumprod()
    risk_dd_60 = (risk_nav - risk_nav.rolling(60).max()) / risk_nav.rolling(60).max()
    variables['v2_risk_dd_60d'] = risk_dd_60.shift(1)
    var_meta['v2_risk_dd_60d'] = {'category': '回撤位置', 'name': '原风险池距60日高点回撤',
                                  'logic': '原池深回撤时大池独有是否更易修复？',
                                  'computable': True, 'future_function': False}

    # 大池独有相对原池的回撤差
    variables['v2_dd_diff_60d'] = variables['v2_bak_dd_60d'] - variables['v2_risk_dd_60d']
    var_meta['v2_dd_diff_60d'] = {'category': '回撤位置', 'name': '大池独有-原池 60日回撤差',
                                  'logic': '大池独有相对原池回撤更深时是否更易修复？',
                                  'computable': True, 'future_function': False}

    # ===== 3. 横截面离散度变量 =====
    # 37 只风险 ETF 横截面 20 日收益离散度（每日截面 std）
    all_ret_20d = (1 + all_ret).rolling(20).apply(np.prod, raw=True) - 1
    xs_disp = all_ret_20d.std(axis=1)
    variables['v3_xs_dispersion_20d'] = xs_disp.shift(1)
    var_meta['v3_xs_dispersion_20d'] = {'category': '横截面离散度', 'name': '37只风险ETF 20日收益横截面离散度',
                                        'logic': '横截面机会多时大池是否更值得开放？',
                                        'computable': True, 'future_function': False}

    # 大池独有内部收益离散度
    bak_disp = all_ret_20d[BAK_ONLY].std(axis=1)
    variables['v3_bak_internal_disp_20d'] = bak_disp.shift(1)
    var_meta['v3_bak_internal_disp_20d'] = {'category': '横截面离散度', 'name': '大池独有内部20日收益离散度',
                                            'logic': '大池独有内部分化时是否更值得开放？',
                                            'computable': True, 'future_function': False}

    # Top ETF 与中位 ETF 的差距（Top1 - median，20日收益）
    top1 = all_ret_20d.max(axis=1)
    median = all_ret_20d.median(axis=1)
    variables['v3_top1_minus_median_20d'] = (top1 - median).shift(1)
    var_meta['v3_top1_minus_median_20d'] = {'category': '横截面离散度', 'name': 'Top1与中位ETF 20日收益差',
                                            'logic': '头部ETF大幅领先时大池是否更值得开放？',
                                            'computable': True, 'future_function': False}

    # ===== 4. 相关性 / 拥挤度变量 =====
    # 大池独有内部平均相关性（60日滚动 pairwise correlation）
    # 计算量较大，用 20 日滚动简化
    bak_ret_df = all_ret[BAK_ONLY]
    # 每日：过去 60 日 ETF 两两相关性的均值
    print("  [计算] 大池独有内部相关性（60日滚动，耗时较长）...")
    n_bak = len(BAK_ONLY)
    rolling_corr_list = []
    for i in range(60, len(bak_ret_df)):
        window = bak_ret_df.iloc[i-60:i]
        corr_mat = window.corr()
        # 取上三角均值（不含对角线）
        upper_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape, dtype=bool), k=1))
        mean_corr = upper_tri.stack().mean()
        rolling_corr_list.append((bak_ret_df.index[i], mean_corr))
    corr_series = pd.Series(dict(rolling_corr_list))
    variables['v4_bak_internal_corr_60d'] = corr_series.shift(1)
    var_meta['v4_bak_internal_corr_60d'] = {'category': '相关性/拥挤度', 'name': '大池独有内部60日平均相关性',
                                            'logic': '高度同涨同跌时大池是否风险更大？',
                                            'computable': True, 'future_function': False}

    # 原风险池内部相关性
    print("  [计算] 原风险池内部相关性（60日滚动）...")
    risk_ret_df = all_ret[ORIGINAL_RISK_POOL]
    rolling_corr_risk = []
    for i in range(60, len(risk_ret_df)):
        window = risk_ret_df.iloc[i-60:i]
        corr_mat = window.corr()
        upper_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape, dtype=bool), k=1))
        mean_corr = upper_tri.stack().mean()
        rolling_corr_risk.append((risk_ret_df.index[i], mean_corr))
    corr_risk_series = pd.Series(dict(rolling_corr_risk))
    variables['v4_risk_internal_corr_60d'] = corr_risk_series.shift(1)
    var_meta['v4_risk_internal_corr_60d'] = {'category': '相关性/拥挤度', 'name': '原风险池内部60日平均相关性',
                                             'logic': '原池高度同涨同跌时是否预示主题拥挤？',
                                             'computable': True, 'future_function': False}

    # ===== 5. 趋势效率 / 噪音变量 =====
    # 大池独有 20 日收益路径效率 = |净收益| / sum(|日收益|)
    bak_roll_net = bak_ret.rolling(20).apply(lambda x: x.sum(), raw=True)
    bak_roll_abs_sum = bak_ret.rolling(20).apply(lambda x: np.abs(x).sum(), raw=True)
    efficiency = bak_roll_net.abs() / bak_roll_abs_sum.replace(0, np.nan)
    variables['v5_bak_efficiency_20d'] = efficiency.shift(1)
    var_meta['v5_bak_efficiency_20d'] = {'category': '趋势效率/噪音', 'name': '大池独有20日收益路径效率',
                                         'logic': '少数跳涨的趋势延续性是否较差？',
                                         'computable': True, 'future_function': False}

    # 大池独有 20 日涨跌日比例（上涨天数/总天数）
    bak_up_ratio = bak_ret.rolling(20).apply(lambda x: (x > 0).sum() / len(x), raw=True)
    variables['v5_bak_up_day_ratio_20d'] = bak_up_ratio.shift(1)
    var_meta['v5_bak_up_day_ratio_20d'] = {'category': '趋势效率/噪音', 'name': '大池独有20日上涨日比例',
                                           'logic': '上涨日比例极端时是否预示反转？',
                                           'computable': True, 'future_function': False}

    # 最大单日贡献占比（20日内最大单日收益/绝对累计收益）
    def max_day_contrib(x):
        abs_sum = np.abs(x).sum()
        if abs_sum == 0:
            return np.nan
        return np.abs(x).max() / abs_sum
    bak_max_contrib = bak_ret.rolling(20).apply(max_day_contrib, raw=True)
    variables['v5_bak_max_day_contrib_20d'] = bak_max_contrib.shift(1)
    var_meta['v5_bak_max_day_contrib_20d'] = {'category': '趋势效率/噪音', 'name': '大池独有20日最大单日贡献占比',
                                              'logic': '趋势由少数跳涨贡献时延续性是否较差？',
                                              'computable': True, 'future_function': False}

    # ===== 6. 流动性变量 =====
    # HData 有 amount 字段
    # 大池独有成交额均值（20日滚动）
    bak_amt = amount[BAK_ONLY]
    bak_amt_mean = bak_amt.mean(axis=1).rolling(20).mean()
    variables['v6_bak_amt_mean_20d'] = bak_amt_mean.shift(1)
    var_meta['v6_bak_amt_mean_20d'] = {'category': '流动性', 'name': '大池独有成交额20日均值',
                                       'logic': '主题爆发前是否有成交额提前扩张？',
                                       'computable': True, 'future_function': False}

    # 成交额变化率（当前20日 / 过去60日 - 1）
    bak_amt_60 = bak_amt.mean(axis=1).rolling(60).mean()
    amt_change = bak_amt.mean(axis=1).rolling(20).mean() / bak_amt_60.replace(0, np.nan) - 1
    variables['v6_bak_amt_change'] = amt_change.shift(1)
    var_meta['v6_bak_amt_change'] = {'category': '流动性', 'name': '大池独有成交额变化率(20d/60d-1)',
                                     'logic': '成交额扩张时主题是否更易延续？',
                                     'computable': True, 'future_function': False}

    # 成交额分位数（当前20日均值在过去252日的分位）
    bak_amt_20 = bak_amt.mean(axis=1).rolling(20).mean()
    amt_quantile = bak_amt_20.rolling(252).rank(pct=True)
    variables['v6_bak_amt_quantile'] = amt_quantile.shift(1)
    var_meta['v6_bak_amt_quantile'] = {'category': '流动性', 'name': '大池独有成交额分位数(252日)',
                                       'logic': '成交额处于历史高位时主题是否更易延续？',
                                       'computable': True, 'future_function': False}

    return variables, var_meta


def descriptive_bucketing(variables, var_meta, future_excess_20d, price_dates):
    """对每个候选变量做描述性分桶（qcut 3 桶），观察未来 20 日超额收益

    注意：这是候选观察，不是切换规则。
    """
    observations = {}
    future = future_excess_20d

    for var_name, var_series in variables.items():
        meta = var_meta[var_name]
        # 对齐日期
        df = pd.DataFrame({'var': var_series, 'future': future}).dropna()
        if len(df) < 100:
            observations[var_name] = {
                'meta': meta, 'n_total': int(len(df)),
                'note': '样本不足，跳过分桶',
            }
            continue

        # 检查变量取值分布，若取值过少用直接分组
        unique_vals = df['var'].nunique()
        if unique_vals <= 5:
            # 分类型变量，按值分组
            grouped = df.groupby('var')['future'].agg(['mean', 'count'])
            buckets = {str(k): {'mean': round(v['mean']*100, 2), 'count': int(v['count'])}
                       for k, v in grouped.to_dict('index').items()}
            obs = {'meta': meta, 'n_total': int(len(df)), 'bucketing': 'categorical',
                   'all_sample': buckets}
        else:
            # 连续变量，qcut 3 桶
            try:
                df['bucket'] = pd.qcut(df['var'], 3, labels=['low', 'mid', 'high'], duplicates='drop')
            except Exception:
                observations[var_name] = {'meta': meta, 'n_total': int(len(df)),
                                          'note': 'qcut 失败，跳过'}
                continue
            grouped = df.groupby('bucket')['future'].agg(['mean', 'count'])

            # 全样本
            all_sample = {}
            for b in ['low', 'mid', 'high']:
                if b in grouped.index:
                    all_sample[b] = {'mean': round(grouped.loc[b, 'mean']*100, 2),
                                     'count': int(grouped.loc[b, 'count'])}

            # 按年份拆分
            df['year'] = df.index.year
            yearly = {}
            for yr in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
                yr_df = df[df['year'] == yr]
                if len(yr_df) < 20:
                    yearly[str(yr)] = {'note': f'样本不足(n={len(yr_df)})'}
                    continue
                yr_grouped = yr_df.groupby('bucket')['future'].agg(['mean', 'count'])
                yr_data = {}
                for b in ['low', 'mid', 'high']:
                    if b in yr_grouped.index:
                        yr_data[b] = {'mean': round(yr_grouped.loc[b, 'mean']*100, 2),
                                      'count': int(yr_grouped.loc[b, 'count'])}
                yearly[str(yr)] = yr_data

            obs = {'meta': meta, 'n_total': int(len(df)), 'bucketing': 'qcut3',
                   'all_sample': all_sample, 'yearly': yearly}

        observations[var_name] = obs

    return observations


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("P3-03 核心池/卫星池定义审计 + 非动量候选变量挖掘")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载 HData 日线数据...")
    data = load_daily_data()

    # 2. 子任务 A：ETF 逐只画像
    print("\n[2/6] 子任务 A：ETF 逐只画像...")
    profiles = build_etf_profile(data)
    print(f"  完成: {len(profiles)} 只 ETF 画像")

    # 输出 ETF 画像表
    profile_table = []
    for p in profiles:
        row = {
            'code': p['code'],
            'theme': p['theme'],
            'theme_category': p['theme_category'],
            'is_original_risk': p['is_original_risk'],
            'is_defensive': p['is_defensive'],
            'is_bak_only': p['is_bak_only'],
            'data_start': p['data_start'],
            'data_end': p.get('data_end', ''),
            'n_valid_days': p['n_valid_days'],
            'data_coverage_pct': p['data_coverage'],
            'amt_mean_yuan': p.get('amt_mean_yuan'),
            'amt_median_yuan': p.get('amt_median_yuan'),
            'vol_annual_pct': p.get('vol_annual_pct'),
            'max_dd_pct': p.get('max_dd_pct'),
            'max_gap_days': p.get('max_gap_days'),
            'has_long_gap': p.get('has_long_gap'),
            'single_year_dominant': p.get('single_year_dominant'),
        }
        for yr in ['2020', '2021', '2022', '2023', '2024', '2025', '2026H1']:
            if yr in p.get('yearly', {}):
                row[f'ret_{yr}_pct'] = p['yearly'][yr].get('return_pct')
                row[f'maxdd_{yr}_pct'] = p['yearly'][yr].get('max_dd_pct')
            else:
                row[f'ret_{yr}_pct'] = None
                row[f'maxdd_{yr}_pct'] = None
        profile_table.append(row)

    pd.DataFrame(profile_table).to_csv(OUT_DIR / 'etf_profile_table.csv', index=False, encoding='utf-8-sig')
    with open(OUT_DIR / 'etf_profile_table.json', 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False, default=str)

    # 3. 子任务 B：核心池定义审计
    print("\n[3/6] 子任务 B：核心池定义审计...")
    audit = audit_core_pool(profiles)
    print(f"  核心池保留候选: {len(audit['core_retain'])} 只")
    print(f"  核心池争议候选: {len(audit['core_disputed'])} 只")
    print(f"  卫星池候选: {len(audit['satellite'])} 只")

    # 4. 子任务 C：卫星池主题结构审计
    print("\n[4/6] 子任务 C：卫星池主题结构审计...")
    theme_profiles = build_theme_profile(data, profiles)
    src_2024_2026 = analyze_2024_2026_source(data, profiles)
    print(f"  主题大类: {len(theme_profiles)} 个")
    print(f"  2024 主题收益来源:")
    for cat, ret in sorted(src_2024_2026['2024'].items(), key=lambda x: -(x[1] or -999)):
        print(f"    {cat}: {ret}%")
    print(f"  2026H1 主题收益来源:")
    for cat, ret in sorted(src_2024_2026['2026H1'].items(), key=lambda x: -(x[1] or -999)):
        print(f"    {cat}: {ret}%")

    theme_table = []
    for t in theme_profiles:
        row = {
            'theme_category': t['theme_category'],
            'n_etfs': t['n_etfs'],
            'etfs': ','.join(t['etfs']),
            'n_days': t['n_days'],
            'vol_annual_pct': t['vol_annual_pct'],
            'max_dd_pct': t['max_dd_pct'],
            'amt_mean_yuan': t['amt_mean_yuan'],
            'single_year_dominant': t['single_year_dominant'],
            'dominant_year': t['dominant_year'],
            'is_bak_only_dominant': t['is_bak_only_dominant'],
        }
        for yr in ['2020', '2021', '2022', '2023', '2024', '2025', '2026H1']:
            row[f'ret_{yr}_pct'] = t['yearly'].get(yr, {}).get('return_pct')
        theme_table.append(row)
    pd.DataFrame(theme_table).to_csv(OUT_DIR / 'theme_profile_table.csv', index=False, encoding='utf-8-sig')
    with open(OUT_DIR / 'theme_profile_table.json', 'w', encoding='utf-8') as f:
        json.dump({'themes': theme_profiles, 'source_2024_2026': src_2024_2026}, f, indent=2, ensure_ascii=False, default=str)

    # 5. 子任务 D：非动量候选变量挖掘
    print("\n[5/6] 子任务 D：非动量候选变量挖掘...")
    variables, var_meta = calc_candidate_variables(data)
    print(f"  候选变量数: {len(variables)}")

    # 三层审计之第一层：可计算性审计
    calc_audit = {}
    for var_name, meta in var_meta.items():
        var_series = variables[var_name]
        n_valid = var_series.notna().sum()
        n_total = len(var_series)
        missing_rate = 1 - n_valid / n_total if n_total > 0 else 1
        calc_audit[var_name] = {
            'category': meta['category'],
            'name': meta['name'],
            'causal_logic': meta['logic'],
            'computable': meta['computable'],
            'future_function_risk': meta['future_function'],
            'n_valid': int(n_valid),
            'n_total': int(n_total),
            'missing_rate_pct': round(missing_rate * 100, 2),
            't_minus_1': True,  # 所有变量都 shift(1)
        }

    with open(OUT_DIR / 'candidate_variables.json', 'w', encoding='utf-8') as f:
        json.dump(calc_audit, f, indent=2, ensure_ascii=False)

    # 三层审计之第三层：描述性分桶观察
    print("\n[6/6] 候选变量描述性分桶观察（未来20日超额收益）...")
    future_excess_20d = calc_future_excess_return(data, 20)
    observations = descriptive_bucketing(variables, var_meta, future_excess_20d, data['close'].index)

    with open(OUT_DIR / 'candidate_variable_observations.json', 'w', encoding='utf-8') as f:
        json.dump(observations, f, indent=2, ensure_ascii=False, default=str)

    # 打印摘要
    print("\n" + "=" * 60)
    print("候选变量分桶摘要（全样本 20 日未来超额收益 %）")
    print("=" * 60)
    for var_name, obs in observations.items():
        if 'all_sample' not in obs:
            print(f"  {var_name}: {obs.get('note', '无分桶')}")
            continue
        all_s = obs['all_sample']
        if obs.get('bucketing') == 'qcut3':
            low = all_s.get('low', {}).get('mean', 'N/A')
            mid = all_s.get('mid', {}).get('mean', 'N/A')
            high = all_s.get('high', {}).get('mean', 'N/A')
            print(f"  {var_name}: low={low} mid={mid} high={high} (n={obs['n_total']})")
        else:
            parts = [f"{k}={v['mean']}(n={v['count']})" for k, v in all_s.items()]
            print(f"  {var_name}: {' '.join(parts)}")

    print(f"\n输出目录: {OUT_DIR}")
    print("P3-03 脚本完成。请人工撰写 P3-03_POOL_AUDIT_REPORT.md")


if __name__ == '__main__':
    main()
