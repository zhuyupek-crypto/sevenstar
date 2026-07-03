"""
七星高照 母版交易日志比较器
==============================
解析聚宽导出的交易记录（交易记录240101-241231.txt），
与本地 run_local.py 产出的 results/qixing_local_*.json 对比，
输出按"日期+证券+方向"对齐的差异报告。

运行方式:
  python compare_mother_log.py                          # 默认对比 2024 全年
  python compare_mother_log.py 2024-01-01 2024-12-31    # 指定区间
  python compare_mother_log.py 2024-01-01 2024-01-31 results/qixing_local_2024-01-01_2024-01-31.json
"""
import sys
import os
import re
import json
import csv
from pathlib import Path
from collections import defaultdict


# ---- 路径 ----
ROOT = Path(__file__).resolve().parent
JQ_TRADES_FILE = ROOT / '聚宽回测结果' / '交易记录240101-241231.txt'


# ---- 聚宽交易记录解析 ----
# 格式: 2024-01-02\t14:01:00\t纳指ETF(513100.XSHG)\t买\t市价单\t825000股\t1.212\t999,900.00\t0.00\t199.98
# 偶数行（行2,5,8...）是空白分隔行，需跳过
TRADE_LINE_RE = re.compile(
    r'^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+'   # 日期 时间
    r'(.+?)\((\d{6}\.(?:XSHG|XSHE))\)\s+'                # 名称(代码)
    r'(买|卖)\s+'                                        # 方向
    r'(\S+)\s+'                                          # 订单类型
    r'(-?\d+)股\s+'                                      # 数量（含负号）
    r'([\d.]+)\s+'                                       # 价格
    r'(-?[\d,]+\.\d+)\s+'                               # 金额
    r'(-?[\d,]+\.\d+)\s+'                               # 税/盈亏
    r'([\d.]+)$'                                         # 佣金
)


def parse_jq_trades(path):
    """解析聚宽交易记录，返回 list[dict]"""
    trades = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = TRADE_LINE_RE.match(line)
            if not m:
                continue
            date, time, name, code, side, _otype, amount_str, price_str, value_str, tax_str, comm_str = m.groups()
            amount = int(amount_str)
            trades.append({
                'date': date,
                'time': time[:5],  # HH:MM
                'code': code,
                'name': name,
                'action': 'buy' if side == '买' else 'sell',
                'amount': abs(amount),
                'signed_amount': amount,
                'price': float(price_str),
                'value': float(value_str.replace(',', '')),
                'tax': float(tax_str.replace(',', '')),
                'commission': float(comm_string_clean(comm_str)),
            })
    return trades


def comm_string_clean(s):
    return s.replace(',', '')


def load_local_trades(json_path):
    """加载本地 run_local.py 产出的 JSON 交易记录"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    trades = []
    for t in data.get('trades', []):
        amount = int(t.get('amount', 0))
        trades.append({
            'date': t.get('date', ''),
            'time': t.get('time', ''),
            'code': t.get('security', ''),
            'action': 'buy' if amount > 0 else 'sell',
            'amount': abs(amount),
            'signed_amount': amount,
            'price': float(t.get('price', 0)),
            'commission': float(t.get('commission', 0)),
            'tax': float(t.get('tax', 0)),
        })
    return trades


def trade_key(t):
    """对齐键：日期 + 代码 + 方向"""
    return (t['date'], t['code'], t['action'])


def compare_trades(jq_trades, local_trades, tolerance=0.011):
    """
    对齐对比，返回差异列表。
    tolerance: 价格/数量允许的浮点误差。
    """
    jq_by_key = defaultdict(list)
    local_by_key = defaultdict(list)
    for t in jq_trades:
        jq_by_key[trade_key(t)].append(t)
    for t in local_trades:
        local_by_key[trade_key(t)].append(t)

    diffs = []
    all_keys = sorted(set(jq_by_key) | set(local_by_key))
    for k in all_keys:
        jq_list = jq_by_key.get(k, [])
        local_list = local_by_key.get(k, [])
        n = max(len(jq_list), len(local_list))
        for i in range(n):
            j = jq_list[i] if i < len(jq_list) else None
            l = local_list[i] if i < len(local_list) else None
            if j is None:
                diffs.append({
                    'date': k[0], 'code': k[1], 'action': k[2],
                    'side': 'extra_in_local',
                    'jq_time': '', 'local_time': l['time'],
                    'jq_amount': '', 'local_amount': l['amount'],
                    'jq_price': '', 'local_price': l['price'],
                    'jq_comm': '', 'local_comm': l['commission'],
                    'diff_note': '本地多出',
                })
                continue
            if l is None:
                diffs.append({
                    'date': k[0], 'code': k[1], 'action': k[2],
                    'side': 'missing_in_local',
                    'jq_time': j['time'], 'local_time': '',
                    'jq_amount': j['amount'], 'local_amount': '',
                    'jq_price': j['price'], 'local_price': '',
                    'jq_comm': j['commission'], 'local_comm': '',
                    'diff_note': '本地缺失',
                })
                continue

            # 两者都有，比较细节
            notes = []
            amount_diff = j['amount'] - l['amount']
            price_diff = j['price'] - l['price']
            comm_diff = j['commission'] - l['commission']

            if abs(amount_diff) > 0:
                notes.append(f'数量差{amount_diff:+d}')
            if abs(price_diff) > tolerance:
                notes.append(f'价格差{price_diff:+.4f}')
            if abs(comm_diff) > 0.5:
                notes.append(f'佣金差{comm_diff:+.2f}')

            if notes:
                diffs.append({
                    'date': k[0], 'code': k[1], 'action': k[2],
                    'side': 'mismatch',
                    'jq_time': j['time'], 'local_time': l['time'],
                    'jq_amount': j['amount'], 'local_amount': l['amount'],
                    'jq_price': j['price'], 'local_price': l['price'],
                    'jq_comm': j['commission'], 'local_comm': l['commission'],
                    'diff_note': '; '.join(notes),
                })
    return diffs


def filter_by_range(trades, start_date, end_date):
    return [t for t in trades if start_date <= t['date'] <= end_date]


def summarize(diffs, jq_count, local_count):
    """生成差异汇总"""
    by_side = defaultdict(int)
    by_code = defaultdict(int)
    for d in diffs:
        by_side[d['side']] += 1
        by_code[d['code']] += 1

    lines = []
    lines.append('=' * 70)
    lines.append('七星高照 母版对齐差异报告')
    lines.append('=' * 70)
    lines.append(f'聚宽交易笔数: {jq_count}')
    lines.append(f'本地交易笔数: {local_count}')
    lines.append(f'差异记录数:   {len(diffs)}')
    lines.append('')
    lines.append('差异分类:')
    for side, cnt in sorted(by_side.items()):
        lines.append(f'  {side}: {cnt}')
    lines.append('')
    if by_code:
        lines.append('按证券分布:')
        for code, cnt in sorted(by_code.items(), key=lambda x: -x[1])[:10]:
            lines.append(f'  {code}: {cnt}')
    lines.append('')
    return '\n'.join(lines)


def main():
    start_date = sys.argv[1] if len(sys.argv) > 1 else '2024-01-01'
    end_date = sys.argv[2] if len(sys.argv) > 2 else '2024-12-31'
    local_json = sys.argv[3] if len(sys.argv) > 3 else None

    # 自动查找本地 JSON
    if local_json is None:
        candidate = ROOT / 'results' / f'qixing_local_{start_date}_{end_date}.json'
        if candidate.exists():
            local_json = str(candidate)
        else:
            # 找最新的
            results_dir = ROOT / 'results'
            if results_dir.exists():
                jsons = sorted(results_dir.glob('qixing_local_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
                if jsons:
                    local_json = str(jsons[0])
            if not local_json:
                print(f'错误: 未找到本地结果 JSON，请先运行 run_local.py 或显式指定路径')
                sys.exit(1)

    print(f'聚宽交易记录: {JQ_TRADES_FILE}')
    print(f'本地结果JSON: {local_json}')
    print(f'对比区间:     {start_date} ~ {end_date}')
    print()

    jq_trades = parse_jq_trades(JQ_TRADES_FILE)
    jq_trades = filter_by_range(jq_trades, start_date, end_date)

    local_trades = load_local_trades(local_json)
    local_trades = filter_by_range(local_trades, start_date, end_date)

    diffs = compare_trades(jq_trades, local_trades)

    print(summarize(diffs, len(jq_trades), len(local_trades)))

    if diffs:
        print('差异明细 (前50条):')
        print('-' * 110)
        print(f'{"日期":<12}{"代码":<16}{"方向":<5}{"类型":<18}{"聚宽数量":>10}{"本地数量":>10}{"聚宽价":>10}{"本地价":>10}  差异说明')
        print('-' * 110)
        for d in diffs[:50]:
            jq_amt = str(d["jq_amount"]) if d["jq_amount"] != "" else "-"
            loc_amt = str(d["local_amount"]) if d["local_amount"] != "" else "-"
            jq_prc = f'{d["jq_price"]:.4f}' if isinstance(d["jq_price"], (int, float)) and d["jq_price"] != "" else "-"
            loc_prc = f'{d["local_price"]:.4f}' if isinstance(d["local_price"], (int, float)) and d["local_price"] != "" else "-"
            print(f'{d["date"]:<12}{d["code"]:<16}{d["action"]:<5}{d["side"]:<18}'
                  f'{jq_amt:>10}{loc_amt:>10}'
                  f'{jq_prc:>10}{loc_prc:>10}  {d["diff_note"]}')

    # 保存差异到 CSV
    if diffs:
        diff_csv = ROOT / 'results' / f'diff_{start_date}_{end_date}.csv'
        diff_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(diff_csv, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'date', 'code', 'action', 'side',
                'jq_time', 'local_time',
                'jq_amount', 'local_amount',
                'jq_price', 'local_price',
                'jq_comm', 'local_comm',
                'diff_note',
            ])
            writer.writeheader()
            writer.writerows(diffs)
        print(f'\n差异明细已保存: {diff_csv}')

    return 0 if not diffs else 1


if __name__ == '__main__':
    sys.exit(main())
