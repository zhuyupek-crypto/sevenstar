"""
七星高照 P2 Walk-Forward 滚动验证脚本
====================================
在 7 个滚动测试段上分别跑 A/B/C/D 四组实验，验证 D 是否稳定优于 A/B/C。
参数固定为 baseline，不调参。

用法:
  python run_walk_forward.py                    # 跑所有 7 个测试段
  python run_walk_forward.py --window 2020      # 只跑 2020 测试段
  python run_walk_forward.py --window 2020,2021 # 跑 2020 和 2021
  python run_walk_forward.py --skip-existing    # 跳过已有结果
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_local import QixingParityRunner

# ==== Baseline 参数（与 P1 一致，不调参） ====
BASELINE = {
    'lookback_days': 25,
    'holdings_num': 1,
    'short_lookback_days': 10,
    'profit_protection_threshold': 0.05,
    'profit_protection_lookback': 1,
}

# ==== 大ETF池（37只，与 P1 一致） ====
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

INITIAL_CASH = 1000000

# ==== 7 个滚动测试段 ====
WINDOWS = [
    ('2020',    '2020-01-02', '2020-12-31'),
    ('2021',    '2021-01-04', '2021-12-31'),
    ('2022',    '2022-01-04', '2022-12-30'),
    ('2023',    '2023-01-03', '2023-12-29'),
    ('2024',    '2024-01-02', '2024-12-31'),
    ('2025',    '2025-01-02', '2025-12-31'),
    ('2026H1',  '2026-01-02', '2026-06-30'),
]

# ==== 四组实验配置 ====
GROUPS = {
    'A': {'pool': 'default', 'score_mode': 'baseline',     'nav_mode': 'parity'},
    'B': {'pool': 'default', 'score_mode': 'multi_period', 'nav_mode': 'parity'},
    'C': {'pool': 'bak',     'score_mode': 'baseline',     'nav_mode': 'realistic'},
    'D': {'pool': 'bak',     'score_mode': 'multi_period', 'nav_mode': 'realistic'},
}


def compute_metrics(res, label, elapsed):
    """计算收益/回撤/夏普/Calmar 等指标（与 run_sweep.py 一致）"""
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


def run_single(group, window_label, start, end, out_dir, skip_existing=False):
    """运行单组单窗口回测"""
    cfg = GROUPS[group]
    label = f"{group}_{window_label}"
    save_path = out_dir / f"{label}.json"

    if skip_existing and save_path.exists():
        print(f"[SKIP] {label} 已存在，跳过")
        with open(save_path, encoding='utf-8') as f:
            data = json.load(f)
        return data.get('metrics', {})

    print(f"\n{'=' * 70}")
    print(f"[WF] {label}  区间: {start} ~ {end}")
    print(f"[WF] 池: {cfg['pool']}  得分: {cfg['score_mode']}  NAV: {cfg['nav_mode']}")
    print(f"{'=' * 70}")

    overrides = dict(BASELINE)
    if cfg['pool'] == 'bak':
        overrides['etf_pool'] = list(POOL_BAK)

    runner = QixingParityRunner(
        start, end,
        param_overrides=overrides,
        score_mode=cfg['score_mode'],
        nav_mode=cfg['nav_mode'],
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
        json.dump({**res, 'label': label, 'group': group, 'window': window_label,
                   'start': start, 'end': end, 'metrics': metrics},
                  f, indent=2, ensure_ascii=False, default=str)

    print(f"[WF] {label} 完成: 权益={metrics['final_value']} "
          f"收益={metrics['total_return_pct']}% 回撤={metrics['max_drawdown_pct']}% "
          f"夏普={metrics['sharpe']} 耗时={metrics['elapsed_s']}s")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='七星高照 P2 Walk-Forward 滚动验证')
    parser.add_argument('--window', default=None,
                        help='指定测试段（逗号分隔，如 2020,2021），默认全部')
    parser.add_argument('--skip-existing', action='store_true',
                        help='跳过已有结果')
    parser.add_argument('--out', default=None,
                        help='输出目录（默认: qixing_optimize/runs/walk_forward）')
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else (ROOT / 'qixing_optimize' / 'runs' / 'walk_forward')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 选择测试段
    if args.window:
        selected_labels = [w.strip() for w in args.window.split(',')]
        windows = [w for w in WINDOWS if w[0] in selected_labels]
    else:
        windows = WINDOWS

    print(f"\n{'#' * 70}")
    print(f"# P2 Walk-Forward 滚动验证")
    print(f"# 测试段: {[w[0] for w in windows]}")
    print(f"# 实验组: A(7只+baseline) B(7只+multi) C(37只+baseline) D(37只+multi)")
    print(f"# 输出目录: {out_dir}")
    print(f"{'#' * 70}")

    all_results = {}
    for window_label, start, end in windows:
        window_results = {}
        for group in ['A', 'B', 'C', 'D']:
            m = run_single(group, window_label, start, end, out_dir,
                          skip_existing=args.skip_existing)
            window_results[group] = m
        all_results[window_label] = window_results

        # 打印该窗口的汇总
        print(f"\n--- {window_label} 窗口汇总 ---")
        print(f"{'组':<4} {'最终权益':>14} {'总收益%':>9} {'年化%':>8} {'回撤%':>8} {'夏普':>7} {'Calmar':>7}")
        for g in ['A', 'B', 'C', 'D']:
            r = window_results.get(g, {})
            if 'error' in r:
                print(f"{g:<4} ERROR: {r['error']}")
            else:
                print(f"{g:<4} {r.get('final_value', 0):>14,.2f} "
                      f"{r.get('total_return_pct', 0):>9.2f} "
                      f"{r.get('annualized_return_pct', 0):>8.2f} "
                      f"{r.get('max_drawdown_pct', 0):>8.2f} "
                      f"{r.get('sharpe', 0):>7.3f} "
                      f"{r.get('calmar', 0):>7.3f}")

    # 保存汇总
    summary_path = out_dir / 'walk_forward_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n汇总已保存: {summary_path}")

    # 打印完整对比表
    print(f"\n{'=' * 100}")
    print(f"{'窗口':<8} {'A收益%':>8} {'B收益%':>8} {'C收益%':>8} {'D收益%':>8} "
          f"{'A回撤%':>8} {'B回撤%':>8} {'C回撤%':>8} {'D回撤%':>8} "
          f"{'D最优?':>6}")
    print(f"{'-' * 100}")
    for window_label, _, _ in windows:
        wr = all_results.get(window_label, {})
        a = wr.get('A', {})
        b = wr.get('B', {})
        c = wr.get('C', {})
        d = wr.get('D', {})
        a_r = a.get('total_return_pct', 0) if 'error' not in a else 0
        b_r = b.get('total_return_pct', 0) if 'error' not in b else 0
        c_r = c.get('total_return_pct', 0) if 'error' not in c else 0
        d_r = d.get('total_return_pct', 0) if 'error' not in d else 0
        a_d = a.get('max_drawdown_pct', 0) if 'error' not in a else 0
        b_d = b.get('max_drawdown_pct', 0) if 'error' not in b else 0
        c_d = c.get('max_drawdown_pct', 0) if 'error' not in c else 0
        d_d = d.get('max_drawdown_pct', 0) if 'error' not in d else 0
        d_best = '✅' if d_r == max(a_r, b_r, c_r, d_r) else '✗'
        print(f"{window_label:<8} {a_r:>8.2f} {b_r:>8.2f} {c_r:>8.2f} {d_r:>8.2f} "
              f"{a_d:>8.2f} {b_d:>8.2f} {c_d:>8.2f} {d_d:>8.2f} {d_best:>6}")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
