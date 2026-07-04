"""
七星高照 参数敏感性扫描脚本
=============================
支持单变量扫描和多变量矩阵扫描，不修改策略源码。
参数通过 QixingParityRunner.param_overrides 注入 g 对象。

用法:
  python run_sweep.py --mode functional              # 4 runs 季度功能验证
  python run_sweep.py --mode single --var lookback_days --values 15,20,25,30,40
  python run_sweep.py --mode matrix --start 2024-01-01 --end 2024-12-31
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path

import numpy as np

# 项目根目录（qixing_optimize/scripts -> 项目根）
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_local import QixingParityRunner

# ==== Baseline 参数（与策略 initialize 一致） ====
BASELINE = {
    'lookback_days': 25,
    'holdings_num': 1,
    'short_lookback_days': 10,
    'profit_protection_threshold': 0.05,
    'profit_protection_lookback': 1,
}

# ==== 大ETF池（去掉缺失的159201，共37只） ====
POOL_BAK = [
    '518880.XSHG', '159980.XSHE', '159985.XSHE', '501018.XSHG', '161226.XSHE', '159981.XSHE',
    '513100.XSHG', '159509.XSHE', '513290.XSHG', '513500.XSHG', '159529.XSHE', '513400.XSHG',
    '513520.XSHG', '513030.XSHG', '513080.XSHG', '513310.XSHG', '513730.XSHG',
    '159792.XSHE', '513130.XSHG', '513050.XSHG', '159920.XSHE', '513690.XSHG',
    '510300.XSHG', '510500.XSHG', '510050.XSHG', '510210.XSHG', '159915.XSHE', '588080.XSHG',
    '512100.XSHG', '563360.XSHG', '563300.XSHG',
    '512890.XSHG', '159967.XSHE', '512040.XSHG',
    '511380.XSHG', '511010.XSHG', '511220.XSHG',
]

# ==== 全量扫描矩阵 ====
SWEEP_MATRIX = {
    'lookback_days': [15, 20, 25, 30, 40],
    'holdings_num': [1, 2, 3],
    'short_lookback_days': [5, 10, 15],
    'profit_protection_threshold': [0.03, 0.05, 0.07, 0.10],
    'profit_protection_lookback': [1, 3, 5, 10],
}

# ==== 功能验证：每变量 baseline + 1 变体 ====
FUNCTIONAL_VARIANTS = [
    ('lookback_days', 30),
    ('holdings_num', 2),
    ('short_lookback_days', 15),
    ('profit_protection_threshold', 0.07),
    ('profit_protection_lookback', 5),
]

INITIAL_CASH = 1000000


def parse_values(val_str, type_cast=float):
    """解析逗号分隔的值列表"""
    parts = [v.strip() for v in val_str.split(',')]
    if type_cast == int:
        return [int(p) for p in parts]
    return [float(p) for p in parts]


def get_var_type(var_name):
    """根据变量名推断类型"""
    if var_name in ('lookback_days', 'holdings_num', 'short_lookback_days', 'profit_protection_lookback'):
        return int
    return float


def compute_metrics(res, label, elapsed):
    """计算收益/回撤/夏普/Calmar 等指标"""
    daily = res.get('daily', [])
    if not daily:
        return {'label': label, 'final_value': res.get('final_value'), 'error': 'no daily data'}

    values = np.array([d['value'] for d in daily], dtype=float)
    total_return = values[-1] / INITIAL_CASH - 1

    # 最大回撤
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    max_dd = float(np.max(drawdown))

    # 日收益率 → 年化夏普
    daily_returns = np.diff(values) / values[:-1]
    sharpe = 0.0
    if daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * (252 ** 0.5))

    # Calmar = 年化收益率 / 最大回撤（标准定义，不是总收益/最大回撤）
    # 对多年度区间差异显著：总收益/回撤会高估 Calmar
    n_days = len(values)
    if n_days > 0 and total_return > -1:
        annualized_return = (1 + total_return) ** (252.0 / n_days) - 1
    else:
        annualized_return = 0.0
    calmar = float(annualized_return / max_dd) if max_dd > 0 else 0.0

    trades = res.get('trades', [])
    buy_count = sum(1 for t in trades if t['amount'] > 0)
    sell_count = sum(1 for t in trades if t['amount'] < 0)

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
    }


def run_single(overrides, label, start, end, out_dir, pool='default', score_mode='baseline', nav_mode='parity'):
    """运行单次回测并保存结果"""
    print(f"\n{'=' * 60}")
    print(f"[SWEEP] {label}  区间: {start} ~ {end}  池: {pool}  得分: {score_mode}  nav: {nav_mode}")
    print(f"[SWEEP] overrides: {overrides}")
    print(f"{'=' * 60}")

    # 大池注入
    extra_overrides = dict(overrides)
    if pool == 'bak':
        extra_overrides['etf_pool'] = list(POOL_BAK)

    runner = QixingParityRunner(start, end, param_overrides=extra_overrides,
                                score_mode=score_mode, nav_mode=nav_mode)
    t0 = time.monotonic()
    try:
        res = runner.run()
    except Exception as e:
        print(f"!!! 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    elapsed = time.monotonic() - t0

    metrics = compute_metrics(res, label, elapsed)

    # 保存完整结果
    save_path = out_dir / f"{label.replace('=', '_').replace('.', 'p')}.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({**res, 'label': label, 'overrides': overrides, 'metrics': metrics},
                  f, indent=2, ensure_ascii=False, default=str)
    print(f"[SWEEP] {label} 完成: 权益={metrics['final_value']} "
          f"收益={metrics['total_return_pct']}% 回撤={metrics['max_drawdown_pct']}% "
          f"夏普={metrics['sharpe']} 耗时={metrics['elapsed_s']}s")
    return metrics


def print_summary(results):
    """打印汇总表格"""
    print(f"\n{'=' * 105}")
    print(f"{'参数':<25} {'最终权益':>12} {'总收益%':>9} {'年化%':>8} {'回撤%':>8} {'夏普':>7} {'Calmar':>7} {'天数':>6} {'耗时s':>7}")
    print(f"{'-' * 105}")
    for r in results:
        if 'error' in r:
            print(f"{r['label']:<25} {'ERROR':>12} {r.get('error', '')}")
            continue
        print(f"{r['label']:<25} {r['final_value']:>12.2f} {r['total_return_pct']:>9.2f} "
              f"{r['annualized_return_pct']:>8.2f} {r['max_drawdown_pct']:>8.2f} {r['sharpe']:>7.3f} {r['calmar']:>7.3f} "
              f"{r['n_trade_days']:>6} {r['elapsed_s']:>7.1f}")
    print(f"{'=' * 105}")


def mode_functional(args):
    """功能验证：baseline + 4 个变体"""
    out_dir = Path(args.out) / 'functional'
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = args.start, args.end

    results = []
    # baseline
    m = run_single(dict(BASELINE), 'baseline', start, end, out_dir)
    if m:
        results.append(m)
    # 4 个变体
    for var, val in FUNCTIONAL_VARIANTS:
        overrides = dict(BASELINE)
        overrides[var] = val
        label = f"{var}={val}"
        m = run_single(overrides, label, start, end, out_dir)
        if m:
            results.append(m)

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print_summary(results)
    return results


def mode_single(args):
    """单变量扫描"""
    if not args.var or not args.values:
        print("单变量扫描需要 --var 和 --values 参数")
        sys.exit(1)

    type_cast = get_var_type(args.var)
    values = parse_values(args.values, type_cast)
    out_dir = Path(args.out) / f'single_{args.var}'
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    # baseline（如果不在 values 中则单独跑）
    baseline_val = BASELINE.get(args.var)
    if baseline_val not in values:
        m = run_single(dict(BASELINE), f'{args.var}={baseline_val}(baseline)',
                       args.start, args.end, out_dir)
        if m:
            results.append(m)
    # 扫描值
    for val in values:
        overrides = dict(BASELINE)
        overrides[args.var] = val
        label = f"{args.var}={val}"
        m = run_single(overrides, label, args.start, args.end, out_dir)
        if m:
            results.append(m)

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print_summary(results)
    return results


def mode_matrix(args):
    """全量矩阵：4 个变量单独扫描"""
    out_dir = Path(args.out) / 'matrix'
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for var, values in SWEEP_MATRIX.items():
        print(f"\n{'#' * 60}")
        print(f"# 扫描变量: {var}  值: {values}")
        print(f"{'#' * 60}")
        sub_dir = out_dir / var
        sub_dir.mkdir(parents=True, exist_ok=True)
        for val in values:
            overrides = dict(BASELINE)
            overrides[var] = val
            label = f"{var}={val}"
            m = run_single(overrides, label, args.start, args.end, sub_dir)
            if m:
                all_results.append(m)

    with open(out_dir / 'all_summary.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print_summary(all_results)
    return all_results


def mode_pool(args):
    """标的池扩展扫描：default vs bak"""
    out_dir = Path(args.out) / f'pool_{args.pool}'
    out_dir.mkdir(parents=True, exist_ok=True)

    label = f'pool={args.pool}'
    m = run_single(dict(BASELINE), label, args.start, args.end, out_dir,
                   pool=args.pool, nav_mode=args.nav_mode)
    return [m] if m else []


def mode_score(args):
    """得分模式扫描"""
    out_dir = Path(args.out) / f'score_{args.score_mode}_pool_{args.pool}'
    out_dir.mkdir(parents=True, exist_ok=True)

    label = f'score_mode={args.score_mode},pool={args.pool}'
    m = run_single(dict(BASELINE), label, args.start, args.end, out_dir,
                   pool=args.pool, score_mode=args.score_mode, nav_mode=args.nav_mode)
    return [m] if m else []


def main():
    parser = argparse.ArgumentParser(description='七星高照参数敏感性扫描')
    parser.add_argument('--mode', choices=['functional', 'single', 'matrix', 'pool', 'score'],
                        default='functional', help='扫描模式')
    parser.add_argument('--start', default='2024-01-02', help='开始日期')
    parser.add_argument('--end', default='2024-12-31', help='结束日期')
    parser.add_argument('--var', help='单变量扫描的变量名')
    parser.add_argument('--values', help='单变量扫描的值列表（逗号分隔）')
    parser.add_argument('--pool', choices=['default', 'bak'], default='default', help='ETF池')
    parser.add_argument('--score-mode', choices=['baseline', 'multi_period', 'vol_adjusted'],
                        default='baseline', help='得分模式')
    parser.add_argument('--out', default=None, help='输出目录（默认: qixing_optimize/runs）')
    parser.add_argument('--nav-mode', choices=['parity', 'realistic', 'legacy_invalid'],
                        default='parity', help='NAV 缺失处理模式（默认: parity）')
    args = parser.parse_args()

    out_root = Path(args.out) if args.out else (ROOT / 'qixing_optimize' / 'runs')
    out_root.mkdir(parents=True, exist_ok=True)
    args.out = str(out_root)

    if args.mode == 'functional':
        mode_functional(args)
    elif args.mode == 'single':
        mode_single(args)
    elif args.mode == 'matrix':
        mode_matrix(args)
    elif args.mode == 'pool':
        mode_pool(args)
    elif args.mode == 'score':
        mode_score(args)


if __name__ == '__main__':
    main()
