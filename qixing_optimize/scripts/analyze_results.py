"""
七星高照 参数扫描结果分析
===========================
读取 run_sweep.py 的输出，生成 Markdown 报告，判断 baseline 是否在平稳区。

用法:
  python analyze_results.py --dir qixing_optimize/runs/functional
  python analyze_results.py --dir qixing_optimize/runs/matrix --full
"""
import sys
import json
import argparse
from pathlib import Path


def load_results(result_dir):
    """加载扫描结果"""
    result_dir = Path(result_dir)
    # 优先读 summary.json
    summary_path = result_dir / 'summary.json'
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    # 兜底：读 all_summary.json
    all_path = result_dir / 'all_summary.json'
    if all_path.exists():
        with open(all_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    # 兜底：扫描子目录
    results = []
    for sub in sorted(result_dir.iterdir()):
        if sub.is_dir():
            sub_summary = sub / 'summary.json'
            if sub_summary.exists():
                with open(sub_summary, 'r', encoding='utf-8') as f:
                    results.extend(json.load(f))
    return results


def find_baseline(results):
    """找到 baseline 结果"""
    for r in results:
        label = r.get('label', '')
        if 'baseline' in label.lower():
            return r
    # 如果没有显式 baseline，尝试找参数值等于 BASELINE 的
    baseline_vals = {'lookback_days': 25, 'holdings_num': 1,
                     'short_lookback_days': 10, 'profit_protection_threshold': 0.05}
    for r in results:
        label = r.get('label', '')
        for var, val in baseline_vals.items():
            if label == f"{var}={val}":
                return r
    return None


def parse_var_val(label):
    """从 label 解析变量名和值，如 'lookback_days=30' -> ('lookback_days', 30.0)"""
    if '=' not in label:
        return None, None
    parts = label.split('=', 1)
    var = parts[0].strip()
    val_str = parts[1].strip().replace('(baseline)', '')
    try:
        val = float(val_str)
        if val == int(val):
            val = int(val)
    except ValueError:
        val = val_str
    return var, val


def group_by_var(results):
    """按变量名分组"""
    groups = {}
    for r in results:
        var, val = parse_var_val(r.get('label', ''))
        if var is None:
            continue
        if var not in groups:
            groups[var] = []
        groups[var].append({**r, '_val': val})
    # 按值排序
    for var in groups:
        groups[var].sort(key=lambda x: x['_val'] if isinstance(x['_val'], (int, float)) else 0)
    return groups


def generate_report(results, full=False):
    """生成 Markdown 报告"""
    lines = []
    lines.append("# 七星高照 参数敏感性扫描报告\n")

    baseline = find_baseline(results)
    if baseline:
        lines.append("## Baseline 性能\n")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|----|")
        lines.append(f"| 最终权益 | {baseline.get('final_value', 'N/A')} |")
        lines.append(f"| 总收益率 | {baseline.get('total_return_pct', 'N/A')}% |")
        lines.append(f"| 最大回撤 | {baseline.get('max_drawdown_pct', 'N/A')}% |")
        lines.append(f"| 夏普比率 | {baseline.get('sharpe', 'N/A')} |")
        lines.append(f"| Calmar | {baseline.get('calmar', 'N/A')} |")
        lines.append(f"| 交易笔数 | {baseline.get('trade_count', 'N/A')} |")
        lines.append("")

    groups = group_by_var(results)
    if not groups:
        lines.append("## 未找到可分组的变量扫描结果\n")
        return '\n'.join(lines)

    lines.append("## 单变量敏感性分析\n")
    for var, items in groups.items():
        lines.append(f"### {var}\n")
        lines.append(f"| 值 | 最终权益 | 收益% | 回撤% | 夏普 | Calmar | 交易 | 相对baseline |")
        lines.append(f"|----|----------|-------|-------|------|--------|------|-------------|")

        base_return = baseline.get('total_return_pct', 0) if baseline else 0
        for item in items:
            ret = item.get('total_return_pct', 0)
            diff = ret - base_return
            diff_str = f"{diff:+.2f}%" if diff != 0 else "—"
            lines.append(f"| {item['_val']} | {item.get('final_value', 0):.2f} | "
                         f"{ret:.2f} | {item.get('max_drawdown_pct', 0):.2f} | "
                         f"{item.get('sharpe', 0):.3f} | {item.get('calmar', 0):.3f} | "
                         f"{item.get('trade_count', 0)} | {diff_str} |")
        lines.append("")

        # 平稳区判断
        if len(items) >= 2 and baseline:
            base_val = None
            baseline_vals = {'lookback_days': 25, 'holdings_num': 1,
                             'short_lookback_days': 10, 'profit_protection_threshold': 0.05}
            if var in baseline_vals:
                base_val = baseline_vals[var]

            if base_val is not None:
                # 找 baseline 位置
                base_idx = None
                for i, item in enumerate(items):
                    if item['_val'] == base_val:
                        base_idx = i
                        break

                if base_idx is not None:
                    lines.append(f"**平稳区分析** (baseline={base_val}):\n")
                    # 检查邻近值
                    neighbors = []
                    if base_idx > 0:
                        neighbors.append(('左', items[base_idx - 1]))
                    if base_idx < len(items) - 1:
                        neighbors.append(('右', items[base_idx + 1]))

                    is_stable = True
                    for side, nb in neighbors:
                        nb_ret = nb.get('total_return_pct', 0)
                        diff_pct = abs(nb_ret - base_return)
                        status = "平稳" if diff_pct < 5 else "波动"
                        if diff_pct >= 10:
                            is_stable = False
                        lines.append(f"- {side}邻值 {nb['_val']}: 收益差 {nb_ret - base_return:+.2f}% → {status}")
                    lines.append(f"\n**结论**: baseline {'在平稳区' if is_stable else '不在平稳区'}\n")

    # 汇总表格
    lines.append("## 完整结果汇总\n")
    lines.append(f"| 参数 | 最终权益 | 收益% | 回撤% | 夏普 | Calmar | 交易 |")
    lines.append(f"|------|----------|-------|-------|------|--------|------|")
    for r in sorted(results, key=lambda x: x.get('total_return_pct', 0), reverse=True):
        if 'error' in r:
            continue
        lines.append(f"| {r.get('label', 'N/A')} | {r.get('final_value', 0):.2f} | "
                     f"{r.get('total_return_pct', 0):.2f} | {r.get('max_drawdown_pct', 0):.2f} | "
                     f"{r.get('sharpe', 0):.3f} | {r.get('calmar', 0):.3f} | "
                     f"{r.get('trade_count', 0)} |")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='七星高照参数扫描结果分析')
    parser.add_argument('--dir', required=True, help='扫描结果目录')
    parser.add_argument('--full', action='store_true', help='生成完整报告')
    parser.add_argument('--out', default=None, help='输出报告路径（默认打印到stdout）')
    args = parser.parse_args()

    results = load_results(args.dir)
    if not results:
        print(f"未找到结果文件: {args.dir}")
        sys.exit(1)

    report = generate_report(results, full=args.full)

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"报告已保存: {args.out}")
    else:
        print(report)


if __name__ == '__main__':
    main()
