"""
七星高照 P5-00 聚宽母版 vs 本地 O7_baseline 绩效复刻验收
=========================================================
目标：验证本地化后的 O7_baseline 是否真实复刻聚宽母版。

数据来源：
- 聚宽：工作目录/聚宽回测结果/ 下的 3 个文件
  - 每日持仓&资金.txt（日度持仓+总价值）
  - 交易记录240101-241231.txt（2024 全年交易明细）
  - 交易日志.txt（策略日志，仅到 2024-03-21，辅助用）
- 本地：qixing_optimize/runs/p4_01_core_pool_validation/O7_baseline_2024.json

边界：
- 不做任何新策略优化
- 不修改 O7_baseline
- 不调整参数来贴合聚宽结果
- 不新增池
- 不重新启动 O7/C8D 切换研究
- 不用聚宽结果反推规则
- 不宣布可实盘
- 不修改 P0-P4 冻结结论
"""
import sys
import os
import re
import json
import math
from pathlib import Path
from collections import OrderedDict, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
JQ_DIR = ROOT / '聚宽回测结果'
LOCAL_DIR = ROOT / 'qixing_optimize' / 'runs' / 'p4_01_core_pool_validation'
OUT_DIR = ROOT / 'qixing_optimize' / 'runs' / 'p5_00_jq_local_parity_validation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# O7 池定义（P4-01 冻结）
POOL_O7 = [
    "518880.XSHG",  # 黄金
    "159985.XSHE",  # 豆粕
    "501018.XSHG",  # 原油
    "161226.XSHE",  # 白银
    "513100.XSHG",  # 纳指
    "159915.XSHE",  # 创业板
    "511220.XSHG",  # 城投债
]

# 通过标准阈值
NAV_ABS_ERR_THRESHOLD = 0.002      # 逐日净值平均绝对误差 ≤ 0.2%
RETURN_CORR_THRESHOLD = 0.98        # 逐日收益相关系数 ≥ 0.98
TOTAL_RETURN_DIFF_THRESHOLD = 0.01  # 总收益差 ≤ 1pp
MAX_DD_DIFF_THRESHOLD = 0.01        # 最大回撤差 ≤ 1pp
ANNUAL_RETURN_DIFF_THRESHOLD = 0.01 # 年化收益差 ≤ 1pp
TRADE_DATE_MATCH_THRESHOLD = 0.95   # 调仓日期一致率 ≥ 95%
POSITION_MATCH_THRESHOLD = 0.95     # 主要持仓 ETF 一致率 ≥ 95%
FLOAT_EPSILON = 1e-9

# 聚宽交易记录正则（与 compare_mother_log.py 一致）
TRADE_LINE_RE = re.compile(
    r'^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+'
    r'(.+?)\((\d{6}\.(?:XSHG|XSHE))\)\s+'
    r'(买|卖)\s+'
    r'(\S+)\s+'
    r'(-?\d+)股\s+'
    r'([\d.]+)\s+'
    r'(-?[\d,]+\.\d+)\s+'
    r'(-?[\d,]+\.\d+)\s+'
    r'([\d.]+)$'
)

# 每日持仓中的日期行
DATE_LINE_RE = re.compile(r'^(\d{4}-\d{2}-\d{2})$')
# 每日持仓中的总计行
TOTAL_LINE_RE = re.compile(r'总共:([\d,]+\.\d+)')


# ============================================================
# 第一步：聚宽数据清单
# ============================================================
def build_jq_inventory():
    """扫描聚宽数据文件，生成清单"""
    print("[1] 扫描聚宽数据文件 ...")
    inventory = []

    # 扫描 聚宽回测结果 目录
    if JQ_DIR.exists():
        for f in sorted(JQ_DIR.iterdir()):
            if f.is_file():
                stat = f.stat()
                inventory.append({
                    'file_path': str(f),
                    'file_name': f.name,
                    'file_type': f.suffix.lstrip('.'),
                    'file_size_bytes': stat.st_size,
                    'file_size_mb': round(stat.st_size / 1024 / 1024, 4),
                    'readable': True,
                    'used_for_alignment': _determine_usage(f.name),
                })

    # 扫描 聚宽净值导出脚本.py（基金净值导出，非策略 NAV）
    nav_export_script = ROOT / '聚宽净值导出脚本.py'
    if nav_export_script.exists():
        stat = nav_export_script.stat()
        inventory.append({
            'file_path': str(nav_export_script),
            'file_name': nav_export_script.name,
            'file_type': 'py',
            'file_size_bytes': stat.st_size,
            'file_size_mb': round(stat.st_size / 1024 / 1024, 4),
            'readable': True,
            'used_for_alignment': 'no (fund NAV export script, not strategy backtest)',
        })

    # 扫描 jq_diagnose.py
    jq_diag = ROOT / 'jq_diagnose.py'
    if jq_diag.exists():
        stat = jq_diag.stat()
        inventory.append({
            'file_path': str(jq_diag),
            'file_name': jq_diag.name,
            'file_type': 'py',
            'file_size_bytes': stat.st_size,
            'file_size_mb': round(stat.st_size / 1024 / 1024, 4),
            'readable': True,
            'used_for_alignment': 'no (diagnostic script, not data)',
        })

    # 检查用户提到的 zip 文件（预期不存在）
    expected_zip_files = [
        'qixing_nav_export_bundle_20171201_20260703.zip',
        'qixing_nav_processed_bundle.zip',
        '七星高照_聚宽NAV完整导出脚本.py',
    ]
    for zf in expected_zip_files:
        p = ROOT / zf
        inventory.append({
            'file_path': str(p),
            'file_name': zf,
            'file_type': 'missing',
            'file_size_bytes': 0,
            'file_size_mb': 0,
            'readable': False,
            'used_for_alignment': 'no (file not found in working directory)',
        })

    df = pd.DataFrame(inventory)
    df.to_csv(OUT_DIR / 'jq_data_inventory.csv', index=False, encoding='utf-8-sig')
    with open(OUT_DIR / 'jq_data_inventory.json', 'w', encoding='utf-8') as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2)
    print(f"  清单完成：{len(inventory)} 个文件条目")
    return df


def _determine_usage(filename):
    """根据文件名判断用途"""
    if '每日持仓' in filename:
        return 'yes (daily NAV + positions)'
    if '交易记录' in filename:
        return 'yes (trade records)'
    if '交易日志' in filename:
        return 'auxiliary (strategy log, only to 2024-03-21)'
    return 'no'


# ============================================================
# 第二步：解析聚宽每日持仓 → 日 NAV 序列
# ============================================================
def parse_jq_daily_nav():
    """解析聚宽每日持仓&资金.txt，返回 DataFrame[date, total_value, cash, positions_detail]"""
    print("\n[2] 解析聚宽每日持仓&资金.txt ...")
    fpath = JQ_DIR / '每日持仓&资金.txt'

    rows = []
    current_date = None
    current_positions = []

    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 跳过前 5 行表头
    for line in lines[5:]:
        line = line.strip()
        if not line:
            continue

        # 日期行
        m = DATE_LINE_RE.match(line)
        if m:
            # 保存上一个日期的数据
            if current_date is not None:
                rows.append(_build_daily_row(current_date, current_positions))
            current_date = m.group(1)
            current_positions = []
            continue

        # 持仓行（包含 tab 分隔的字段）
        if current_date and '\t' in line:
            current_positions.append(line)

    # 保存最后一个日期
    if current_date is not None:
        rows.append(_build_daily_row(current_date, current_positions))

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"  解析完成：{len(df)} 个交易日")
    print(f"  日期范围：{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    print(f"  初始总价值：{df['total_value'].iloc[0]:,.2f}")
    print(f"  最终总价值：{df['total_value'].iloc[-1]:,.2f}")
    return df


def _build_daily_row(date, positions):
    """从持仓行列表构建日数据行"""
    total_value = None
    cash = None
    held_etfs = []

    for pos_line in positions:
        # 总计行：\t \t \t 总共:xxx \t yyy
        m = TOTAL_LINE_RE.search(pos_line)
        if m:
            total_value = float(m.group(1).replace(',', ''))
            continue

        # Cash 行：Cash\t \t \t -99.98 \t 0.00
        if pos_line.startswith('Cash'):
            parts = pos_line.split('\t')
            for p in parts:
                p = p.strip()
                try:
                    cash = float(p.replace(',', ''))
                    break
                except (ValueError, TypeError):
                    continue
            continue

        # ETF 持仓行：纳指ETF(513100.XSHG)\t825000股\t1.21\t998,250.00\t-1,650.00
        # 提取 ETF 代码
        code_match = re.search(r'(\d{6}\.(?:XSHG|XSHE))', pos_line)
        if code_match:
            held_etfs.append(code_match.group(1))

    return {
        'date': date,
        'total_value': total_value,
        'cash': cash,
        'held_etfs': '|'.join(held_etfs) if held_etfs else '',
        'n_held': len(held_etfs),
    }


# ============================================================
# 第三步：解析聚宽交易记录
# ============================================================
def parse_jq_trades():
    """解析聚宽交易记录，返回 list[dict]"""
    print("\n[3] 解析聚宽交易记录240101-241231.txt ...")
    fpath = JQ_DIR / '交易记录240101-241231.txt'

    trades = []
    with open(fpath, 'r', encoding='utf-8') as f:
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
                'time': time[:5],
                'code': code,
                'name': name,
                'action': 'buy' if side == '买' else 'sell',
                'amount': abs(amount),
                'signed_amount': amount,
                'price': float(price_str),
                'value': float(value_str.replace(',', '')),
                'tax': float(tax_str.replace(',', '')),
                'commission': float(comm_str),
            })

    print(f"  解析完成：{len(trades)} 笔交易")
    if trades:
        print(f"  日期范围：{trades[0]['date']} ~ {trades[-1]['date']}")
        total_comm = sum(t['commission'] for t in trades)
        print(f"  总佣金：{total_comm:,.2f}")
    return trades


# ============================================================
# 第四步：加载本地 O7_baseline_2024
# ============================================================
def load_local_baseline():
    """加载本地 O7_baseline_2024.json"""
    print("\n[4] 加载本地 O7_baseline_2024.json ...")
    fpath = LOCAL_DIR / 'O7_baseline_2024.json'
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # daily
    daily = data['daily']
    df = pd.DataFrame(daily)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df.rename(columns={'value': 'total_value'})

    # trades
    trades = data['trades']
    for t in trades:
        t['action'] = 'buy' if t['amount'] > 0 else 'sell'
        t['amount'] = abs(t['amount'])
        t['signed_amount'] = t['amount'] if t['action'] == 'buy' else -t['amount']
        t['code'] = t.pop('security')
        t['name'] = ''
        t['value'] = t['price'] * t['signed_amount']
        t['time'] = t.get('time', '')

    # metrics
    metrics = data['metrics']

    print(f"  daily: {len(df)} 个交易日")
    print(f"  日期范围：{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    print(f"  初始总价值：{df['total_value'].iloc[0]:,.2f}")
    print(f"  最终总价值：{df['total_value'].iloc[-1]:,.2f}")
    print(f"  trades: {len(trades)} 笔")
    print(f"  metrics.total_return_pct: {metrics.get('total_return_pct')}")
    return df, trades, metrics, data


# ============================================================
# 第五步：对齐口径检查
# ============================================================
def check_alignment(jq_nav, local_nav, jq_trades, local_trades, local_raw):
    """对齐口径检查"""
    print("\n[5] 对齐口径检查 ...")
    checks = []

    # 1. 起止日期
    jq_start = jq_nav['date'].iloc[0]
    jq_end = jq_nav['date'].iloc[-1]
    local_start = local_nav['date'].iloc[0]
    local_end = local_nav['date'].iloc[-1]
    date_aligned = (jq_start == local_start) and (jq_end == local_end)
    checks.append({
        'item': '起止日期',
        'jq_value': f'{jq_start.date()} ~ {jq_end.date()}',
        'local_value': f'{local_start.date()} ~ {local_end.date()}',
        'aligned': date_aligned,
        'note': '' if date_aligned else '日期不完全一致',
    })

    # 2. 初始资金（首日总价值）
    jq_init = jq_nav['total_value'].iloc[0]
    local_init = local_nav['total_value'].iloc[0]
    init_diff = abs(jq_init - local_init)
    init_aligned = init_diff < 1.0
    checks.append({
        'item': '初始总价值（首日）',
        'jq_value': f'{jq_init:,.2f}',
        'local_value': f'{local_init:,.2f}',
        'aligned': init_aligned,
        'note': f'差值 {init_diff:,.2f}' if not init_aligned else '完全一致',
    })

    # 3. 交易日数
    jq_n_days = len(jq_nav)
    local_n_days = len(local_nav)
    days_aligned = jq_n_days == local_n_days
    checks.append({
        'item': '交易日数',
        'jq_value': str(jq_n_days),
        'local_value': str(local_n_days),
        'aligned': days_aligned,
        'note': '' if days_aligned else f'差 {abs(jq_n_days - local_n_days)} 天',
    })

    # 4. ETF 池
    jq_etfs_in_trades = set(t['code'] for t in jq_trades)
    local_etfs_in_trades = set(t['code'] for t in local_trades)
    pool_aligned = jq_etfs_in_trades == local_etfs_in_trades
    checks.append({
        'item': '交易 ETF 池',
        'jq_value': '|'.join(sorted(jq_etfs_in_trades)),
        'local_value': '|'.join(sorted(local_etfs_in_trades)),
        'aligned': pool_aligned,
        'note': '' if pool_aligned else f'聚宽独有 {jq_etfs_in_trades - local_etfs_in_trades}, 本地独有 {local_etfs_in_trades - jq_etfs_in_trades}',
    })

    # 5. 成本口径（从首笔交易佣金推断）
    if jq_trades and local_trades:
        jq_first = jq_trades[0]
        local_first = local_trades[0]
        jq_comm_rate = jq_first['commission'] / (jq_first['price'] * jq_first['amount']) if jq_first['price'] * jq_first['amount'] > 0 else 0
        local_comm_rate = local_first['commission'] / (local_first['price'] * local_first['amount']) if local_first['price'] * local_first['amount'] > 0 else 0
        comm_aligned = abs(jq_comm_rate - local_comm_rate) < 0.0001
        checks.append({
            'item': '佣金费率（首笔交易推断）',
            'jq_value': f'{jq_comm_rate:.6f} (佣金{jq_first["commission"]:.2f}/金额{jq_first["price"]*jq_first["amount"]:.2f})',
            'local_value': f'{local_comm_rate:.6f} (佣金{local_first["commission"]:.2f}/金额{local_first["price"]*local_first["amount"]:.2f})',
            'aligned': comm_aligned,
            'note': '' if comm_aligned else '费率不同',
        })

    # 6. 调仓频率（从交易日期推断）
    jq_trade_dates = sorted(set(t['date'] for t in jq_trades))
    local_trade_dates = sorted(set(t['date'] for t in local_trades))
    checks.append({
        'item': '调仓日期数',
        'jq_value': f'{len(jq_trade_dates)} 个调仓日',
        'local_value': f'{len(local_trade_dates)} 个调仓日',
        'aligned': len(jq_trade_dates) == len(local_trade_dates),
        'note': '' if len(jq_trade_dates) == len(local_trade_dates) else f'差 {abs(len(jq_trade_dates) - len(local_trade_dates))} 个调仓日',
    })

    # 7. 复权方式（从价格对比推断）
    if jq_trades and local_trades:
        # 对比前 5 笔交易的价格
        n_compare = min(5, len(jq_trades), len(local_trades))
        price_diffs = []
        for i in range(n_compare):
            jq_p = jq_trades[i]['price']
            local_p = local_trades[i]['price']
            if jq_p > 0:
                price_diffs.append(abs(jq_p - local_p) / jq_p)
        avg_price_diff = sum(price_diffs) / len(price_diffs) if price_diffs else 0
        fq_aligned = avg_price_diff < 0.001
        checks.append({
            'item': '复权方式（前5笔交易价格差异）',
            'jq_value': f'前{n_compare}笔平均价格差 {avg_price_diff:.6f}',
            'local_value': f'前{n_compare}笔平均价格差 {avg_price_diff:.6f}',
            'aligned': fq_aligned,
            'note': '' if fq_aligned else '价格差异较大，可能复权方式不同',
        })

    df = pd.DataFrame(checks)
    print(f"  对齐检查完成：{sum(c['aligned'] for c in checks)}/{len(checks)} 项一致")
    return df, checks


# ============================================================
# 第六步：逐日净值对比
# ============================================================
def compare_nav(jq_nav, local_nav):
    """逐日净值对比"""
    print("\n[6] 逐日净值对比 ...")

    # 合并对齐
    jq = jq_nav[['date', 'total_value']].rename(columns={'total_value': 'jq_nav'})
    local = local_nav[['date', 'total_value']].rename(columns={'total_value': 'local_nav'})
    merged = pd.merge(jq, local, on='date', how='outer').sort_values('date').reset_index(drop=True)

    # 归一化为净值（初始=1.0）
    jq_init = merged['jq_nav'].dropna().iloc[0]
    local_init = merged['local_nav'].dropna().iloc[0]
    merged['jq_nav_norm'] = merged['jq_nav'] / jq_init
    merged['local_nav_norm'] = merged['local_nav'] / local_init

    # 误差
    merged['nav_abs_diff'] = (merged['jq_nav_norm'] - merged['local_nav_norm']).abs()
    merged['nav_pct_diff'] = merged['nav_abs_diff'] / merged['jq_nav_norm'].where(merged['jq_nav_norm'] != 0, 1)

    # 统计
    valid = merged.dropna(subset=['jq_nav_norm', 'local_nav_norm'])
    stats = {
        'n_aligned_days': len(valid),
        'max_abs_diff': float(valid['nav_abs_diff'].max()),
        'mean_abs_diff': float(valid['nav_abs_diff'].mean()),
        'median_abs_diff': float(valid['nav_abs_diff'].median()),
        'max_pct_diff': float(valid['nav_pct_diff'].max()),
        'n_days_gt_0_1pct': int((valid['nav_pct_diff'] > 0.001).sum()),
        'n_days_gt_0_5pct': int((valid['nav_pct_diff'] > 0.005).sum()),
        'n_days_gt_1pct': int((valid['nav_pct_diff'] > 0.01).sum()),
        'jq_final_nav': float(valid['jq_nav_norm'].iloc[-1]),
        'local_final_nav': float(valid['local_nav_norm'].iloc[-1]),
        'final_nav_diff': float(valid['jq_nav_norm'].iloc[-1] - valid['local_nav_norm'].iloc[-1]),
    }

    print(f"  对齐天数：{stats['n_aligned_days']}")
    print(f"  最大绝对误差：{stats['max_abs_diff']:.6f}")
    print(f"  平均绝对误差：{stats['mean_abs_diff']:.6f}")
    print(f"  中位绝对误差：{stats['median_abs_diff']:.6f}")
    print(f"  最大百分比误差：{stats['max_pct_diff']:.6f} ({stats['max_pct_diff']*100:.4f}%)")
    print(f"  误差 >0.1% 天数：{stats['n_days_gt_0_1pct']}")
    print(f"  误差 >0.5% 天数：{stats['n_days_gt_0_5pct']}")
    print(f"  误差 >1.0% 天数：{stats['n_days_gt_1pct']}")
    print(f"  聚宽最终净值：{stats['jq_final_nav']:.6f}")
    print(f"  本地最终净值：{stats['local_final_nav']:.6f}")
    print(f"  最终净值差：{stats['final_nav_diff']:+.6f}")

    # 保存
    out = merged[['date', 'jq_nav', 'local_nav', 'jq_nav_norm', 'local_nav_norm', 'nav_abs_diff', 'nav_pct_diff']]
    out.to_csv(OUT_DIR / 'jq_local_nav_comparison.csv', index=False, encoding='utf-8-sig')
    with open(OUT_DIR / 'jq_local_nav_comparison.json', 'w', encoding='utf-8') as f:
        json.dump({'stats': stats, 'daily_records': out.to_dict(orient='records')}, f, ensure_ascii=False, indent=2, default=str)

    return merged, stats


# ============================================================
# 第七步：逐日收益对比
# ============================================================
def compare_returns(jq_nav, local_nav):
    """逐日收益对比"""
    print("\n[7] 逐日收益对比 ...")

    jq = jq_nav[['date', 'total_value']].rename(columns={'total_value': 'jq_nav'}).copy()
    local = local_nav[['date', 'total_value']].rename(columns={'total_value': 'local_nav'}).copy()

    jq['jq_return'] = jq['jq_nav'].pct_change()
    local['local_return'] = local['local_nav'].pct_change()

    merged = pd.merge(jq[['date', 'jq_return']], local[['date', 'local_return']], on='date', how='inner')
    merged['return_diff'] = merged['jq_return'] - merged['local_return']

    valid = merged.dropna(subset=['jq_return', 'local_return'])
    corr = valid['jq_return'].corr(valid['local_return'])
    direction_match = (np.sign(valid['jq_return']) == np.sign(valid['local_return'])).mean()

    stats = {
        'n_days': len(valid),
        'max_return_diff': float(valid['return_diff'].abs().max()),
        'mean_return_diff': float(valid['return_diff'].abs().mean()),
        'median_return_diff': float(valid['return_diff'].abs().median()),
        'correlation': float(corr) if not np.isnan(corr) else None,
        'direction_match_rate': float(direction_match),
    }

    print(f"  对齐天数：{stats['n_days']}")
    print(f"  最大日收益差：{stats['max_return_diff']:.6f}")
    print(f"  平均日收益差：{stats['mean_return_diff']:.6f}")
    print(f"  相关系数：{stats['correlation']:.6f}")
    print(f"  方向一致率：{stats['direction_match_rate']:.4f} ({stats['direction_match_rate']*100:.2f}%)")

    merged.to_csv(OUT_DIR / 'jq_local_return_comparison.csv', index=False, encoding='utf-8-sig')
    with open(OUT_DIR / 'jq_local_return_comparison.json', 'w', encoding='utf-8') as f:
        json.dump({'stats': stats, 'daily_records': merged.to_dict(orient='records')}, f, ensure_ascii=False, indent=2, default=str)

    return merged, stats


# ============================================================
# 第八步：汇总绩效对比
# ============================================================
def compare_metrics(jq_nav, local_nav, local_metrics):
    """汇总绩效对比"""
    print("\n[8] 汇总绩效对比 ...")

    def compute_metrics(nav_series, label):
        nav = nav_series.dropna().values
        if len(nav) < 2:
            return {}
        total_return = nav[-1] / nav[0] - 1
        n_days = len(nav)
        annual_return = (1 + total_return) ** (252 / n_days) - 1

        # 最大回撤
        peak = np.maximum.accumulate(nav)
        drawdown = (nav - peak) / peak
        max_dd = float(drawdown.min())

        # 夏普（日收益）
        daily_ret = np.diff(nav) / nav[:-1]
        if daily_ret.std() > 0:
            sharpe = float(daily_ret.mean() / daily_ret.std() * math.sqrt(252))
        else:
            sharpe = 0.0

        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0

        return {
            'label': label,
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'calmar': float(calmar),
            'final_nav': float(nav[-1]),
            'init_nav': float(nav[0]),
            'n_days': n_days,
        }

    jq_metrics = compute_metrics(jq_nav['total_value'], 'jq_mother')
    local_comp_metrics = compute_metrics(local_nav['total_value'], 'local_o7_baseline')

    # 从 local_metrics 取 P4-01 原始 metrics
    p4_01_metrics = {
        'label': 'local_o7_baseline_p4_01_raw',
        'total_return': float(local_metrics.get('total_return_pct', 0)) / 100,
        'annual_return': float(local_metrics.get('annualized_return_pct', 0)) / 100,
        'max_drawdown': float(local_metrics.get('max_drawdown_pct', 0)) / 100,
        'sharpe': float(local_metrics.get('sharpe', 0)),
        'calmar': float(local_metrics.get('calmar', 0)),
        'final_nav': float(local_metrics.get('final_value', 0)),
        'n_days': int(local_metrics.get('n_trade_days', 0)),
    }

    rows = []
    for m in [jq_metrics, local_comp_metrics, p4_01_metrics]:
        rows.append(m)

    df = pd.DataFrame(rows)

    # 差异
    diff = {
        'total_return_diff': jq_metrics['total_return'] - local_comp_metrics['total_return'],
        'annual_return_diff': jq_metrics['annual_return'] - local_comp_metrics['annual_return'],
        'max_drawdown_diff': jq_metrics['max_drawdown'] - local_comp_metrics['max_drawdown'],
        'sharpe_diff': jq_metrics['sharpe'] - local_comp_metrics['sharpe'],
        'final_nav_diff': jq_metrics['final_nav'] - local_comp_metrics['final_nav'],
    }

    print(f"  聚宽总收益：{jq_metrics['total_return']*100:.2f}%")
    print(f"  本地总收益：{local_comp_metrics['total_return']*100:.2f}%")
    print(f"  总收益差：{diff['total_return_diff']*100:+.2f}pp")
    print(f"  聚宽最大回撤：{jq_metrics['max_drawdown']*100:.2f}%")
    print(f"  本地最大回撤：{local_comp_metrics['max_drawdown']*100:.2f}%")
    print(f"  回撤差：{diff['max_drawdown_diff']*100:+.2f}pp")
    print(f"  聚宽夏普：{jq_metrics['sharpe']:.4f}")
    print(f"  本地夏普：{local_comp_metrics['sharpe']:.4f}")

    df.to_csv(OUT_DIR / 'jq_local_metric_comparison.csv', index=False, encoding='utf-8-sig')
    with open(OUT_DIR / 'jq_local_metric_comparison.json', 'w', encoding='utf-8') as f:
        json.dump({'metrics': rows, 'diff': diff}, f, ensure_ascii=False, indent=2, default=str)

    return df, diff


# ============================================================
# 第九步：调仓日期与交易对比
# ============================================================
def compare_trades(jq_trades, local_trades):
    """调仓日期与交易对比"""
    print("\n[9] 调仓日期与交易对比 ...")

    jq_trade_dates = sorted(set(t['date'] for t in jq_trades))
    local_trade_dates = sorted(set(t['date'] for t in local_trades))
    jq_set = set(jq_trade_dates)
    local_set = set(local_trade_dates)

    common = jq_set & local_set
    jq_only = jq_set - local_set
    local_only = local_set - jq_set

    # 调仓日期一致率
    union = jq_set | local_set
    date_match_rate = len(common) / len(union) if union else 0

    # 逐笔交易对比（按 date+code+action 对齐）
    def trade_key(t):
        return (t['date'], t['code'], t['action'])

    jq_by_key = defaultdict(list)
    local_by_key = defaultdict(list)
    for t in jq_trades:
        jq_by_key[trade_key(t)].append(t)
    for t in local_trades:
        local_by_key[trade_key(t)].append(t)

    trade_diffs = []
    all_keys = sorted(set(jq_by_key) | set(local_by_key))
    n_matched = 0
    n_mismatched = 0
    n_jq_only = 0
    n_local_only = 0

    for k in all_keys:
        jq_list = jq_by_key.get(k, [])
        local_list = local_by_key.get(k, [])
        n = max(len(jq_list), len(local_list))
        for i in range(n):
            j = jq_list[i] if i < len(jq_list) else None
            l = local_list[i] if i < len(local_list) else None
            if j is None:
                trade_diffs.append({
                    'date': k[0], 'code': k[1], 'action': k[2],
                    'type': 'extra_in_local',
                    'jq_amount': '', 'local_amount': l['amount'],
                    'jq_price': '', 'local_price': l['price'],
                    'jq_comm': '', 'local_comm': l['commission'],
                    'diff_note': '本地多出',
                })
                n_local_only += 1
            elif l is None:
                trade_diffs.append({
                    'date': k[0], 'code': k[1], 'action': k[2],
                    'type': 'missing_in_local',
                    'jq_amount': j['amount'], 'local_amount': '',
                    'jq_price': j['price'], 'local_price': '',
                    'jq_comm': j['commission'], 'local_comm': '',
                    'diff_note': '本地缺失',
                })
                n_jq_only += 1
            else:
                amount_diff = j['amount'] - l['amount']
                price_diff = j['price'] - l['price']
                comm_diff = j['commission'] - l['commission']
                notes = []
                if abs(amount_diff) > 0:
                    notes.append(f'数量差{amount_diff:+d}')
                if abs(price_diff) > 0.011:
                    notes.append(f'价格差{price_diff:+.4f}')
                if abs(comm_diff) > 0.5:
                    notes.append(f'佣金差{comm_diff:+.2f}')
                if notes:
                    trade_diffs.append({
                        'date': k[0], 'code': k[1], 'action': k[2],
                        'type': 'mismatch',
                        'jq_amount': j['amount'], 'local_amount': l['amount'],
                        'jq_price': j['price'], 'local_price': l['price'],
                        'jq_comm': j['commission'], 'local_comm': l['commission'],
                        'diff_note': '; '.join(notes),
                    })
                    n_mismatched += 1
                else:
                    n_matched += 1

    stats = {
        'jq_n_trades': len(jq_trades),
        'local_n_trades': len(local_trades),
        'jq_n_trade_dates': len(jq_trade_dates),
        'local_n_trade_dates': len(local_trade_dates),
        'n_common_dates': len(common),
        'n_jq_only_dates': len(jq_only),
        'n_local_only_dates': len(local_only),
        'date_match_rate': float(date_match_rate),
        'n_matched_trades': n_matched,
        'n_mismatched_trades': n_mismatched,
        'n_jq_only_trades': n_jq_only,
        'n_local_only_trades': n_local_only,
    }

    print(f"  聚宽交易笔数：{stats['jq_n_trades']}")
    print(f"  本地交易笔数：{stats['local_n_trades']}")
    print(f"  聚宽调仓日数：{stats['jq_n_trade_dates']}")
    print(f"  本地调仓日数：{stats['local_n_trade_dates']}")
    print(f"  调仓日期一致率：{stats['date_match_rate']:.4f} ({stats['date_match_rate']*100:.2f}%)")
    print(f"  完全匹配交易：{stats['n_matched_trades']}")
    print(f"  数值不匹配：{stats['n_mismatched_trades']}")
    print(f"  聚宽独有：{stats['n_jq_only_trades']}")
    print(f"  本地独有：{stats['n_local_only_trades']}")

    # 聚宽独有调仓日（前20个）
    if jq_only:
        print(f"  聚宽独有调仓日（前10）：{sorted(jq_only)[:10]}")
    if local_only:
        print(f"  本地独有调仓日（前10）：{sorted(local_only)[:10]}")

    df = pd.DataFrame(trade_diffs)
    df.to_csv(OUT_DIR / 'jq_local_trade_comparison.csv', index=False, encoding='utf-8-sig')
    with open(OUT_DIR / 'jq_local_trade_comparison.json', 'w', encoding='utf-8') as f:
        json.dump({'stats': stats, 'trade_diffs': trade_diffs}, f, ensure_ascii=False, indent=2, default=str)

    return df, stats, sorted(jq_only), sorted(local_only)


# ============================================================
# 第十步：持仓 ETF 对比
# ============================================================
def compare_positions(jq_nav, local_trades, jq_trades):
    """持仓 ETF 对比（本地从交易推断持仓）"""
    print("\n[10] 持仓 ETF 对比 ...")

    # 聚宽持仓：从 jq_nav 的 held_etfs 字段
    jq_pos = jq_nav[['date', 'held_etfs']].copy()
    jq_pos['jq_held_set'] = jq_pos['held_etfs'].apply(lambda x: set(x.split('|')) if x else set())

    # 本地持仓：从交易记录推断
    local_holding = {}  # code -> amount
    local_daily_pos = {}
    local_trade_by_date = defaultdict(list)
    for t in local_trades:
        local_trade_by_date[t['date']].append(t)

    all_dates = sorted(set(t['date'] for t in local_trades) | set(jq_pos['date'].dt.strftime('%Y-%m-%d').tolist()))

    for d in all_dates:
        d_str = d if isinstance(d, str) else d.strftime('%Y-%m-%d')
        if d_str in local_trade_by_date:
            for t in local_trade_by_date[d_str]:
                code = t['code']
                if t['action'] == 'buy':
                    local_holding[code] = local_holding.get(code, 0) + t['amount']
                else:
                    local_holding[code] = local_holding.get(code, 0) - t['amount']
                    if local_holding[code] <= 0:
                        local_holding.pop(code, None)
        local_daily_pos[d_str] = set(local_holding.keys())

    # 对齐对比
    jq_pos['date_str'] = jq_pos['date'].dt.strftime('%Y-%m-%d')
    position_diffs = []
    n_match = 0
    n_mismatch = 0
    n_days = 0

    for _, row in jq_pos.iterrows():
        d = row['date_str']
        jq_set = row['jq_held_set']
        local_set = local_daily_pos.get(d, set())
        if not jq_set and not local_set:
            continue
        n_days += 1
        if jq_set == local_set:
            n_match += 1
        else:
            n_mismatch += 1
            position_diffs.append({
                'date': d,
                'jq_held': '|'.join(sorted(jq_set)),
                'local_held': '|'.join(sorted(local_set)),
                'jq_only': '|'.join(sorted(jq_set - local_set)),
                'local_only': '|'.join(sorted(local_set - jq_set)),
                'note': f'jq={len(jq_set)}etfs local={len(local_set)}etfs',
            })

    match_rate = n_match / n_days if n_days > 0 else 0

    stats = {
        'n_days_compared': n_days,
        'n_match': n_match,
        'n_mismatch': n_mismatch,
        'position_match_rate': float(match_rate),
    }

    print(f"  对比天数：{stats['n_days_compared']}")
    print(f"  完全匹配：{stats['n_match']}")
    print(f"  不匹配：{stats['n_mismatch']}")
    print(f"  持仓一致率：{stats['position_match_rate']:.4f} ({stats['position_match_rate']*100:.2f}%)")

    df = pd.DataFrame(position_diffs)
    df.to_csv(OUT_DIR / 'jq_local_position_comparison.csv', index=False, encoding='utf-8-sig')
    with open(OUT_DIR / 'jq_local_position_comparison.json', 'w', encoding='utf-8') as f:
        json.dump({'stats': stats, 'position_diffs': position_diffs}, f, ensure_ascii=False, indent=2, default=str)

    return df, stats


# ============================================================
# 第十一步：交易成本对比
# ============================================================
def compare_costs(jq_trades, local_trades):
    """交易成本对比"""
    print("\n[11] 交易成本对比 ...")

    jq_total_comm = sum(t['commission'] for t in jq_trades)
    local_total_comm = sum(t['commission'] for t in local_trades)
    jq_total_tax = sum(t.get('tax', 0) for t in jq_trades)
    local_total_tax = sum(t.get('tax', 0) for t in local_trades)

    jq_total_value = sum(abs(t.get('value', t['price'] * t['amount'])) for t in jq_trades)
    local_total_value = sum(abs(t.get('value', t['price'] * t['amount'])) for t in local_trades)

    stats = {
        'jq_total_commission': float(jq_total_comm),
        'local_total_commission': float(local_total_comm),
        'commission_diff': float(jq_total_comm - local_total_comm),
        'jq_total_tax': float(jq_total_tax),
        'local_total_tax': float(local_total_tax),
        'tax_diff': float(jq_total_tax - local_total_tax),
        'jq_total_trade_value': float(jq_total_value),
        'local_total_trade_value': float(local_total_value),
        'jq_comm_rate': float(jq_total_comm / jq_total_value) if jq_total_value > 0 else 0,
        'local_comm_rate': float(local_total_comm / local_total_value) if local_total_value > 0 else 0,
    }

    print(f"  聚宽总佣金：{stats['jq_total_commission']:,.2f}")
    print(f"  本地总佣金：{stats['local_total_commission']:,.2f}")
    print(f"  佣金差：{stats['commission_diff']:+,.2f}")
    print(f"  聚宽佣金率：{stats['jq_comm_rate']:.6f}")
    print(f"  本地佣金率：{stats['local_comm_rate']:.6f}")

    return stats


# ============================================================
# 第十二步：决策汇总
# ============================================================
def build_decision_summary(nav_stats, return_stats, metric_diff, trade_stats, pos_stats, cost_stats, alignment_checks):
    """决策汇总"""
    print("\n[12] 决策汇总 ...")

    # 通过标准检查
    criteria = {
        '1_date_param_aligned': all(c['aligned'] for c in alignment_checks[:3]),  # 起止日期+初始资金+交易日数
        '2_nav_alignable': nav_stats['n_aligned_days'] > 0,
        '3_total_return_diff_le_1pp': abs(metric_diff['total_return_diff']) <= TOTAL_RETURN_DIFF_THRESHOLD,
        '4_max_dd_diff_le_1pp': abs(metric_diff['max_drawdown_diff']) <= MAX_DD_DIFF_THRESHOLD,
        '5_annual_return_diff_le_1pp': abs(metric_diff['annual_return_diff']) <= ANNUAL_RETURN_DIFF_THRESHOLD,
        '6_nav_mean_abs_err_le_0_2pct': nav_stats['mean_abs_diff'] <= NAV_ABS_ERR_THRESHOLD,
        '7_return_corr_ge_0_98': return_stats['correlation'] is not None and return_stats['correlation'] >= RETURN_CORR_THRESHOLD,
        '8_trade_date_match_ge_95pct': trade_stats['date_match_rate'] >= TRADE_DATE_MATCH_THRESHOLD,
        '9_position_match_ge_95pct': pos_stats['position_match_rate'] >= POSITION_MATCH_THRESHOLD,
        '10_no_systematic_drift': nav_stats['n_days_gt_1pct'] < 10,  # 误差>1%的天数不超过10天
    }

    all_pass = all(criteria.values())

    # 净值级复刻通过条件
    nav_level_pass = all([
        criteria['2_nav_alignable'],
        criteria['3_total_return_diff_le_1pp'],
        criteria['4_max_dd_diff_le_1pp'],
        criteria['5_annual_return_diff_le_1pp'],
        criteria['6_nav_mean_abs_err_le_0_2pct'],
        criteria['7_return_corr_ge_0_98'],
    ])

    # 交易级复刻通过条件
    trade_level_pass = all([
        criteria['8_trade_date_match_ge_95pct'],
        criteria['9_position_match_ge_95pct'],
    ])

    if all_pass:
        final_recommendation = 'parity_passed_proceed_to_live_packaging'
    elif nav_level_pass and not trade_level_pass:
        final_recommendation = 'nav_level_passed_trade_level_failed'
    elif not nav_level_pass:
        final_recommendation = 'parity_failed_fix_localization_first'
    else:
        final_recommendation = 'parity_passed_with_caveats'

    allow_live_packaging = all_pass

    summary = {
        'pass_criteria': criteria,
        'all_pass': all_pass,
        'nav_level_pass': nav_level_pass,
        'trade_level_pass': trade_level_pass,
        'final_recommendation': final_recommendation,
        'allow_live_packaging': allow_live_packaging,
        'nav_stats': nav_stats,
        'return_stats': return_stats,
        'metric_diff': metric_diff,
        'trade_stats': trade_stats,
        'position_stats': pos_stats,
        'cost_stats': cost_stats,
        'data_coverage': {
            'jq_data_dates': '2024-01-02 ~ 2024-12-31 (only 2024)',
            'jq_data_files': 3,
            'jq_trade_log_incomplete': True,
            'jq_trade_log_end_date': '2024-03-21',
            'full_range_parity_not_possible': True,
            'full_range_note': '聚宽数据只覆盖 2024 年，无法做 2020-2026H1 全区间复刻验收',
        },
    }

    print(f"  通过标准：{sum(criteria.values())}/{len(criteria)} 项通过")
    print(f"  净值级复刻：{'通过' if nav_level_pass else '不通过'}")
    print(f"  交易级复刻：{'通过' if trade_level_pass else '不通过'}")
    print(f"  最终建议：{final_recommendation}")
    print(f"  允许实盘化封板：{allow_live_packaging}")

    with open(OUT_DIR / 'p5_00_decision_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    return summary


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("P5-00 聚宽母版 vs 本地 O7_baseline 绩效复刻验收")
    print("=" * 70)

    # 1. 数据清单
    inventory = build_jq_inventory()

    # 2. 解析聚宽数据
    jq_nav = parse_jq_daily_nav()
    jq_trades = parse_jq_trades()

    # 3. 加载本地数据
    local_nav, local_trades, local_metrics, local_raw = load_local_baseline()

    # 4. 对齐口径检查
    alignment_df, alignment_checks = check_alignment(jq_nav, local_nav, jq_trades, local_trades, local_raw)

    # 5. 逐日净值对比
    nav_merged, nav_stats = compare_nav(jq_nav, local_nav)

    # 6. 逐日收益对比
    return_merged, return_stats = compare_returns(jq_nav, local_nav)

    # 7. 汇总绩效对比
    metric_df, metric_diff = compare_metrics(jq_nav, local_nav, local_metrics)

    # 8. 调仓日期与交易对比
    trade_diff_df, trade_stats, jq_only_dates, local_only_dates = compare_trades(jq_trades, local_trades)

    # 9. 持仓 ETF 对比
    pos_diff_df, pos_stats = compare_positions(jq_nav, local_trades, jq_trades)

    # 10. 交易成本对比
    cost_stats = compare_costs(jq_trades, local_trades)

    # 11. 决策汇总
    summary = build_decision_summary(nav_stats, return_stats, metric_diff, trade_stats, pos_stats, cost_stats, alignment_checks)

    print("\n" + "=" * 70)
    print("P5-00 验收完成")
    print("=" * 70)
    print(f"输出目录：{OUT_DIR}")
    print(f"最终建议：{summary['final_recommendation']}")
    print(f"允许实盘化封板：{summary['allow_live_packaging']}")


if __name__ == '__main__':
    main()
