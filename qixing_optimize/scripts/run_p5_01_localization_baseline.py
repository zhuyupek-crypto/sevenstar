"""
P5-01 本地化母版基线回测
=========================
用 run_local.py 的 QixingParityRunner 跑母版策略+O7池，2019-12-05 ~ 2026-06-12。
作为 P5-01 修复防御ETF触发逻辑和交易数量计算逻辑前的对比基线。

边界：
- score_mode='baseline'（母版原版得分逻辑，不做任何 P1-P4 优化）
- nav_mode='parity'（NAV 缺失立即终止，严格对齐聚宽）
- 不修改 O7_baseline
- 不做策略优化
- 不调参
"""
import sys
import os
import json
import time
from pathlib import Path

# 复用 run_local.py 的 QixingParityRunner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from run_local import QixingParityRunner

START = '2019-12-06'  # 159985 NAV 从 2019-12-05 起，首日前一交易日需有 NAV
END = '2026-06-12'

OUT_DIR = Path(__file__).resolve().parents[2] / 'qixing_optimize' / 'runs' / 'p5_01_localization_baseline'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print(f"P5-01 本地化母版基线回测")
    print(f"区间: {START} ~ {END}")
    print(f"配置: 母版策略 + O7池 + score_mode=baseline + nav_mode=parity")
    print(f"输出: {OUT_DIR}")
    print("=" * 70)

    runner = QixingParityRunner(
        start_date=START,
        end_date=END,
        score_mode='baseline',
        nav_mode='parity',
    )

    t0 = time.monotonic()
    try:
        results = runner.run()
    except Exception as e:
        print(f"\n!!! 回测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.monotonic() - t0
    results['elapsed_seconds'] = round(elapsed, 2)

    # 保存完整结果
    out_file = OUT_DIR / f'localization_baseline_{START}_{END}.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n完整结果已保存: {out_file}")

    # 打印摘要
    print_summary(results, elapsed)

    # 保存摘要
    save_summary(results, elapsed)


def print_summary(results, elapsed):
    """打印回测摘要"""
    daily = results.get('daily', [])
    trades = results.get('trades', [])
    final_value = results.get('final_value', 0)
    initial = 1000000

    print("\n" + "=" * 70)
    print("回测摘要")
    print("=" * 70)
    print(f"交易日数: {len(daily)}")
    print(f"交易笔数: {len(trades)}")
    print(f"初始资金: {initial:,.2f}")
    print(f"最终权益: {final_value:,.2f}")
    print(f"总收益: {(final_value/initial - 1)*100:.2f}%")
    print(f"耗时: {elapsed:.1f}s")

    if daily:
        # 计算绩效指标
        navs = [d['value'] for d in daily]
        max_nav = 0
        max_dd = 0
        for v in navs:
            if v > max_nav:
                max_nav = v
            dd = (v - max_nav) / max_nav
            if dd < max_dd:
                max_dd = dd

        years = len(daily) / 250
        total_ret = final_value / initial - 1
        ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

        print(f"年化收益: {ann_ret*100:.2f}%")
        print(f"最大回撤: {max_dd*100:.2f}%")
        print(f"Calmar: {ann_ret / abs(max_dd) if max_dd < 0 else 0:.2f}")

    # NAV 报告
    nav_report = results.get('nav_report', {})
    if nav_report.get('total_missing', 0) > 0:
        print(f"\nNAV 缺失: {nav_report['total_missing']} 次")
        print(f"受影响 ETF: {nav_report.get('affected_pool', [])}")


def save_summary(results, elapsed):
    """保存绩效摘要到 JSON"""
    daily = results.get('daily', [])
    trades = results.get('trades', [])
    final_value = results.get('final_value', 0)
    initial = 1000000

    summary = {
        'config': {
            'start_date': START,
            'end_date': END,
            'score_mode': 'baseline',
            'nav_mode': 'parity',
            'initial_cash': initial,
            'etf_pool': 'O7 (母版默认 7 只)',
            'defensive_etf': '511880.XSHG',
        },
        'basic': {
            'trade_days': len(daily),
            'trade_count': len(trades),
            'final_value': final_value,
            'elapsed_seconds': round(elapsed, 1),
        },
    }

    if daily:
        import numpy as np
        navs = np.array([d['value'] for d in daily])
        rets = np.diff(navs) / navs[:-1]

        max_nav = np.maximum.accumulate(navs)
        dd = (navs - max_nav) / max_nav
        max_dd = dd.min()

        years = len(daily) / 250
        total_ret = final_value / initial - 1
        ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

        sharpe = (np.mean(rets) / np.std(rets) * np.sqrt(250)) if np.std(rets) > 0 else 0
        calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0

        # 找最大回撤区间
        max_dd_end_idx = np.argmin(dd)
        max_dd_start_idx = np.argmax(navs[:max_dd_end_idx + 1]) if max_dd_end_idx > 0 else 0

        summary['metrics'] = {
            'total_return_pct': round(total_ret * 100, 2),
            'annual_return_pct': round(ann_ret * 100, 2),
            'max_drawdown_pct': round(max_dd * 100, 2),
            'max_dd_start': daily[max_dd_start_idx]['date'] if max_dd_start_idx < len(daily) else None,
            'max_dd_end': daily[max_dd_end_idx]['date'] if max_dd_end_idx < len(daily) else None,
            'sharpe': round(sharpe, 3),
            'calmar': round(calmar, 3),
            'volatility_pct': round(np.std(rets) * np.sqrt(250) * 100, 2),
        }

    summary['nav_report'] = results.get('nav_report', {})

    out = OUT_DIR / 'baseline_summary.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"摘要已保存: {out}")


if __name__ == '__main__':
    main()
