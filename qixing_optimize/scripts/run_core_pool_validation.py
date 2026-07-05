"""
P4-01 核心池重定义 walk-forward 验证
=========================================
基准提交：238b797（P3-03 池审计完成）

任务边界：
- 只验证"重新定义核心池，是否能比原风险池 7 只更稳？"
- 不做动态切换，不调参，不引入新择时规则
- 不使用 v2/v6/v3 做切换规则
- B37 引用 P2 已有结果，不重跑

候选池：
  O7:  原风险池 7 只（基准）
  C5:  审计核心 5 只（黄金+创业板+沪深300+中证500+上证50）
  C7D: 审计核心 + 纳指 + 城投债 = 7 只
  C8D: 审计核心 + 纳指 + 城投债 + 豆粕 = 8 只
  B37: 大池 37 只（仅参考，引用 P2）

得分模式：baseline / multi_period
窗口：P2 walk-forward 7 个窗口（2020-2026H1）
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# 项目根目录
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_local import QixingParityRunner

# ==== 输出目录 ====
OUT_DIR = ROOT / 'qixing_optimize' / 'runs' / 'p4_01_core_pool_validation'
OUT_DIR.mkdir(parents=True, exist_ok=True)
WF_DIR = ROOT / 'qixing_optimize' / 'runs' / 'walk_forward'  # P2 结果引用

# ==== 固定参数（P2/P3 已冻结，不得调参） ====
BASELINE = {
    'lookback_days': 25,
    'holdings_num': 1,
    'short_lookback_days': 10,
    'profit_protection_threshold': 0.05,
    'profit_protection_lookback': 1,
    'defensive_etf': '511880.XSHG',
}

# ==== 候选池定义（P4-01 任务书） ====
POOL_O7 = [
    '518880.XSHG',   # 黄金
    '159985.XSHE',   # 豆粕
    '501018.XSHG',   # 原油
    '161226.XSHE',   # 白银
    '513100.XSHG',   # 纳指
    '159915.XSHE',   # 创业板
    '511220.XSHG',   # 城投债
]

POOL_C5 = [
    '518880.XSHG',   # 黄金（原池核心）
    '159915.XSHE',   # 创业板（原池核心）
    '510300.XSHG',   # 沪深300（大池独有核心）
    '510500.XSHG',   # 中证500（大池独有核心）
    '510050.XSHG',   # 上证50（大池独有核心）
]

POOL_C7D = [
    '518880.XSHG',   # 黄金
    '159915.XSHE',   # 创业板
    '510300.XSHG',   # 沪深300
    '510500.XSHG',   # 中证500
    '510050.XSHG',   # 上证50
    '513100.XSHG',   # 纳指（跨资产分散）
    '511220.XSHG',   # 城投债（跨资产分散）
]

POOL_C8D = [
    '518880.XSHG',   # 黄金
    '159915.XSHE',   # 创业板
    '510300.XSHG',   # 沪深300
    '510500.XSHG',   # 中证500
    '510050.XSHG',   # 上证50
    '513100.XSHG',   # 纳指
    '511220.XSHG',   # 城投债
    '159985.XSHE',   # 豆粕（商品分散）
]

# B37 仅引用 P2 结果，不重跑
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

# ==== 测试窗口（P2 walk-forward 冻结） ====
WINDOWS = {
    '2020':   ('2020-01-02', '2020-12-31'),
    '2021':   ('2021-01-04', '2021-12-31'),
    '2022':   ('2022-01-04', '2022-12-30'),
    '2023':   ('2023-01-03', '2023-12-29'),
    '2024':   ('2024-01-02', '2024-12-31'),
    '2025':   ('2025-01-02', '2025-12-31'),
    '2026H1': ('2026-01-02', '2026-06-30'),
}

INITIAL_CASH = 1000000


def compute_metrics(res, label, elapsed):
    """计算收益/回撤/夏普/Calmar 等指标"""
    daily = res.get('daily', [])
    if not daily:
        return {'label': label, 'final_value': res.get('final_value'), 'error': 'no daily data'}

    values = np.array([d['value'] for d in daily], dtype=float)
    total_return = values[-1] / INITIAL_CASH - 1

    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    max_dd = float(np.max(drawdown))

    daily_returns = np.diff(values) / values[:-1]
    sharpe = 0.0
    if daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * (252 ** 0.5))

    n_days = len(values)
    if n_days > 0 and total_return > -1:
        annualized_return = (1 + total_return) ** (252.0 / n_days) - 1
    else:
        annualized_return = 0.0
    calmar = float(annualized_return / max_dd) if max_dd > 0 else 0.0

    trades = res.get('trades', [])
    buy_count = sum(1 for t in trades if t['amount'] > 0)
    sell_count = sum(1 for t in trades if t['amount'] < 0)

    # NAV 审计
    nav_audit = res.get('nav_audit', {})
    fail_open = nav_audit.get('fail_open', False)
    missing_skip = nav_audit.get('missing_skip', False)

    return {
        'label': label,
        'final_value': round(float(values[-1]), 2),
        'total_return_pct': round(total_return * 100, 2),
        'annualized_return_pct': round(annualized_return * 100, 2),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'sharpe': round(sharpe, 3),
        'calmar': round(calmar, 3),
        'n_trade_days': int(n_days),
        'trade_count': len(trades),
        'buy_count': buy_count,
        'sell_count': sell_count,
        'elapsed_s': round(elapsed, 1),
        'nav_fail_open': fail_open,
        'nav_missing_skip': missing_skip,
    }


def run_single(pool_code, pool_list, score_mode, window_label, start, end, skip_existing=True):
    """运行单次回测"""
    label = f"{pool_code}_{score_mode}_{window_label}"
    save_path = OUT_DIR / f"{label}.json"

    if skip_existing and save_path.exists():
        print(f"[SKIP] {label} 已存在")
        with open(save_path, encoding='utf-8') as f:
            data = json.load(f)
        return data.get('metrics', {})

    print(f"\n{'=' * 70}")
    print(f"[P4-01] {label}  区间: {start} ~ {end}")
    print(f"[P4-01] 池: {pool_code} ({len(pool_list)} 只)  得分: {score_mode}  NAV: realistic")
    print(f"{'=' * 70}")

    overrides = dict(BASELINE)
    overrides['etf_pool'] = list(pool_list)

    runner = QixingParityRunner(
        start, end,
        param_overrides=overrides,
        score_mode=score_mode,
        nav_mode='realistic',  # 与 P2 C/D 一致
    )
    t0 = time.monotonic()
    try:
        res = runner.run()
    except Exception as e:
        print(f"!!! {label} 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return {'label': label, 'error': str(e)}
    elapsed = time.monotonic() - t0

    metrics = compute_metrics(res, label, elapsed)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({**res, 'label': label, 'pool_code': pool_code,
                   'score_mode': score_mode, 'window': window_label,
                   'start': start, 'end': end, 'metrics': metrics},
                  f, indent=2, ensure_ascii=False, default=str)

    print(f"[P4-01] {label} 完成: 权益={metrics['final_value']} "
          f"收益={metrics['total_return_pct']}% 回撤={metrics['max_drawdown_pct']}% "
          f"夏普={metrics['sharpe']} 耗时={metrics['elapsed_s']}s")
    return metrics


def load_p2_b37_results():
    """引用 P2 已有 B37 结果（C/D 组对应 B37 baseline/multi_period）"""
    print("\n[引用] P2 B37 结果（C=baseline, D=multi_period）")
    p2_summary_path = WF_DIR / 'walk_forward_summary.json'
    if not p2_summary_path.exists():
        print(f"  [警告] P2 汇总文件不存在: {p2_summary_path}")
        return {}

    with open(p2_summary_path, encoding='utf-8') as f:
        p2_summary = json.load(f)

    b37_results = {}
    for window_label, groups in p2_summary.items():
        # C 组 = B37 + baseline, D 组 = B37 + multi_period
        if 'C' in groups and 'error' not in groups['C']:
            c = groups['C']
            b37_results[f"B37_baseline_{window_label}"] = {
                'label': f"B37_baseline_{window_label}",
                'pool_code': 'B37',
                'score_mode': 'baseline',
                'window': window_label,
                'source': 'P2_C',
                'final_value': c.get('final_value'),
                'total_return_pct': c.get('total_return_pct'),
                'annualized_return_pct': c.get('annualized_return_pct'),
                'max_drawdown_pct': c.get('max_drawdown_pct'),
                'sharpe': c.get('sharpe'),
                'calmar': c.get('calmar'),
                'n_trade_days': c.get('n_trade_days'),
                'trade_count': c.get('trade_count'),
                'buy_count': c.get('buy_count'),
                'sell_count': c.get('sell_count'),
                'elapsed_s': c.get('elapsed_s'),
                'nav_fail_open': False,  # P2 已通过 NAV 预检
                'nav_missing_skip': False,
            }
        if 'D' in groups and 'error' not in groups['D']:
            d = groups['D']
            b37_results[f"B37_multi_period_{window_label}"] = {
                'label': f"B37_multi_period_{window_label}",
                'pool_code': 'B37',
                'score_mode': 'multi_period',
                'window': window_label,
                'source': 'P2_D',
                'final_value': d.get('final_value'),
                'total_return_pct': d.get('total_return_pct'),
                'annualized_return_pct': d.get('annualized_return_pct'),
                'max_drawdown_pct': d.get('max_drawdown_pct'),
                'sharpe': d.get('sharpe'),
                'calmar': d.get('calmar'),
                'n_trade_days': d.get('n_trade_days'),
                'trade_count': d.get('trade_count'),
                'buy_count': d.get('buy_count'),
                'sell_count': d.get('sell_count'),
                'elapsed_s': d.get('elapsed_s'),
                'nav_fail_open': False,
                'nav_missing_skip': False,
            }
    print(f"  引用 B37 结果: {len(b37_results)} 条")
    return b37_results


def aggregate_summary(all_results):
    """跨窗口汇总"""
    # 按 pool_code + score_mode 分组
    groups = defaultdict(list)
    for label, m in all_results.items():
        if 'error' in m:
            continue
        pool_code = m.get('pool_code', label.split('_')[0])
        score_mode = m.get('score_mode', label.split('_')[1])
        window = m.get('window', label.split('_')[-1])
        key = f"{pool_code}_{score_mode}"
        groups[key].append({**m, 'window': window})

    summary = {}
    for key, items in groups.items():
        returns = [it['total_return_pct'] for it in items if it.get('total_return_pct') is not None]
        drawdowns = [it['max_drawdown_pct'] for it in items if it.get('max_drawdown_pct') is not None]
        sharpes = [it['sharpe'] for it in items if it.get('sharpe') is not None]
        trade_counts = [it['trade_count'] for it in items if it.get('trade_count') is not None]

        # 2026H1 表现
        h1_2026 = next((it for it in items if it['window'] == '2026H1'), None)

        # 最优窗口数（该组在各窗口中 total_return 最高的次数）
        n_windows = len(items)
        optimal_count = 0
        for it in items:
            w = it['window']
            # 找该窗口所有组中的最大收益
            max_ret = max(
                (x['total_return_pct'] for x in items if x['window'] == w and x.get('total_return_pct') is not None),
                default=None
            )
            if max_ret is not None and it.get('total_return_pct') == max_ret:
                optimal_count += 1

        # 回撤 ≤ 20% 窗口数
        dd_le_20 = sum(1 for dd in drawdowns if dd <= 20.0)

        summary[key] = {
            'n_windows': n_windows,
            'avg_return_pct': round(float(np.mean(returns)), 2) if returns else None,
            'median_return_pct': round(float(np.median(returns)), 2) if returns else None,
            'worst_return_pct': round(float(np.min(returns)), 2) if returns else None,
            'best_return_pct': round(float(np.max(returns)), 2) if returns else None,
            'max_drawdown_pct': round(float(np.max(drawdowns)), 2) if drawdowns else None,
            'avg_sharpe': round(float(np.mean(sharpes)), 3) if sharpes else None,
            'avg_trade_count': round(float(np.mean(trade_counts)), 1) if trade_counts else None,
            'n_windows_dd_le_20': dd_le_20,
            'n_windows_optimal': optimal_count,
            'return_2026H1_pct': h1_2026.get('total_return_pct') if h1_2026 else None,
            'drawdown_2026H1_pct': h1_2026.get('max_drawdown_pct') if h1_2026 else None,
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description='P4-01 核心池重定义 walk-forward 验证')
    parser.add_argument('--window', default=None,
                        help='指定测试段（逗号分隔，如 2020,2021），默认全部')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='跳过已有结果（默认开启）')
    parser.add_argument('--no-skip', action='store_true',
                        help='强制重跑（覆盖已有结果）')
    args = parser.parse_args()

    skip_existing = args.skip_existing and not args.no_skip

    # 测试段
    if args.window:
        windows = {k: v for k, v in WINDOWS.items() if k in args.window.split(',')}
    else:
        windows = WINDOWS

    # 候选池配置
    pool_configs = [
        ('O7',  POOL_O7),
        ('C5',  POOL_C5),
        ('C7D', POOL_C7D),
        ('C8D', POOL_C8D),
    ]
    score_modes = ['baseline', 'multi_period']

    print("=" * 70)
    print("P4-01 核心池重定义 walk-forward 验证")
    print("=" * 70)
    print(f"候选池: {[p[0] for p in pool_configs]} + B37(引用P2)")
    print(f"得分模式: {score_modes}")
    print(f"窗口: {list(windows.keys())}")
    print(f"总回测数: {len(pool_configs) * len(score_modes) * len(windows)} (B37引用P2)")

    # 运行回测
    all_results = {}
    for pool_code, pool_list in pool_configs:
        for score_mode in score_modes:
            for window_label, (start, end) in windows.items():
                m = run_single(pool_code, pool_list, score_mode,
                               window_label, start, end, skip_existing)
                label = f"{pool_code}_{score_mode}_{window_label}"
                all_results[label] = {**m, 'pool_code': pool_code,
                                       'score_mode': score_mode,
                                       'window': window_label}

    # 引用 P2 B37 结果
    b37_results = load_p2_b37_results()
    all_results.update(b37_results)

    # 保存 window_results
    print("\n[保存] window_results.json/csv")
    window_rows = []
    for label, m in all_results.items():
        if 'error' in m:
            continue
        window_rows.append({
            'label': label,
            'pool_code': m.get('pool_code'),
            'score_mode': m.get('score_mode'),
            'window': m.get('window'),
            'final_value': m.get('final_value'),
            'total_return_pct': m.get('total_return_pct'),
            'annualized_return_pct': m.get('annualized_return_pct'),
            'max_drawdown_pct': m.get('max_drawdown_pct'),
            'sharpe': m.get('sharpe'),
            'calmar': m.get('calmar'),
            'n_trade_days': m.get('n_trade_days'),
            'trade_count': m.get('trade_count'),
            'buy_count': m.get('buy_count'),
            'sell_count': m.get('sell_count'),
            'nav_fail_open': m.get('nav_fail_open'),
            'nav_missing_skip': m.get('nav_missing_skip'),
            'source': m.get('source', 'P4-01'),
        })

    with open(OUT_DIR / 'window_results.json', 'w', encoding='utf-8') as f:
        json.dump(window_rows, f, indent=2, ensure_ascii=False, default=str)

    df_windows = pd.DataFrame(window_rows)
    df_windows.to_csv(OUT_DIR / 'window_results.csv', index=False, encoding='utf-8-sig')

    # 跨窗口汇总
    print("\n[汇总] 跨窗口统计")
    summary = aggregate_summary(all_results)

    with open(OUT_DIR / 'core_pool_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    summary_rows = []
    for key, s in summary.items():
        pool_code, score_mode = key.rsplit('_', 1)
        row = {'pool_code': pool_code, 'score_mode': score_mode, **s}
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_DIR / 'core_pool_summary.csv', index=False, encoding='utf-8-sig')

    # pool_comparison_matrix: 池 × 窗口 收益矩阵
    print("\n[生成] pool_comparison_matrix.csv")
    pivot = df_windows.pivot_table(
        index=['pool_code', 'score_mode'],
        columns='window',
        values='total_return_pct',
        aggfunc='first'
    )
    pivot.to_csv(OUT_DIR / 'pool_comparison_matrix.csv', encoding='utf-8-sig')

    # NAV 审计汇总
    nav_audit = {
        'total_runs': len(window_rows),
        'fail_open_count': sum(1 for r in window_rows if r.get('nav_fail_open')),
        'missing_skip_count': sum(1 for r in window_rows if r.get('nav_missing_skip')),
        'note': 'P4-01 使用 realistic NAV 模式（与 P2 C/D 一致），NAV 缺失时跳过该 ETF',
    }
    with open(OUT_DIR / 'nav_audit_summary.json', 'w', encoding='utf-8') as f:
        json.dump(nav_audit, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print("\n" + "=" * 100)
    print("P4-01 汇总（跨窗口）")
    print("=" * 100)
    print(f"{'组别':<22} {'平均%':>8} {'中位%':>8} {'最差%':>8} {'最大回撤%':>10} "
          f"{'回撤≤20窗口':>12} {'最优窗口':>8} {'2026H1%':>8} {'2026H1回撤%':>12}")
    print("-" * 100)
    for key, s in sorted(summary.items()):
        print(f"{key:<22} {s['avg_return_pct']:>8.2f} {s['median_return_pct']:>8.2f} "
              f"{s['worst_return_pct']:>8.2f} {s['max_drawdown_pct']:>10.2f} "
              f"{s['n_windows_dd_le_20']:>12} {s['n_windows_optimal']:>8} "
              f"{s.get('return_2026H1_pct', 'N/A'):>8} "
              f"{s.get('drawdown_2026H1_pct', 'N/A'):>12}")

    print(f"\n输出目录: {OUT_DIR}")
    print("P4-01 脚本完成。请人工撰写 P4-01_CORE_POOL_VALIDATION_REPORT.md")


if __name__ == '__main__':
    main()
