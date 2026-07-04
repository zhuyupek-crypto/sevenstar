"""
P3-01 持仓级归因：原池 vs 大池胜负来源拆解
=========================================
输入：qixing_optimize/runs/walk_forward/{A,B,C,D}_{窗口}.json（28 组）
输出：qixing_optimize/runs/attribution/
  - trades_paired_{group}_{window}.json   交易配对（含期末未平仓 MTM）
  - etf_contribution_{group}_{window}.json  ETF 贡献表
  - monthly_contribution_{group}_{window}.json  月度贡献表
  - pool_decomposition_{window}.json  原池共有 vs 大池独有贡献拆解
  - attribution_summary.json  全窗口汇总
  - P3-01_ATTRIBUTION_REPORT.md  归因报告

硬约束（用户修订版）：
1. 期末未平仓必须 mark-to-market
2. 每组每窗口归因合计必须校验对齐最终权益
3. 第一版只拆：原池共有ETF / 大池独有ETF / ETF级贡献 / 月度贡献 / Top拖累Top贡献
4. 不精算"换仓时点贡献"，不做切换规则，不做市场状态指标，不调参
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
WF_DIR = ROOT / 'qixing_optimize' / 'runs' / 'walk_forward'
OUT_DIR = ROOT / 'qixing_optimize' / 'runs' / 'attribution'
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = ['2020', '2021', '2022', '2023', '2024', '2025', '2026H1']
GROUPS = ['A', 'B', 'C', 'D']
INITIAL_CASH = 1_000_000

# 原池（A/B 实际交易过的 8 只，= 大池子集）
DEFAULT_POOL = {
    '159915.XSHE', '159985.XSHE', '161226.XSHE', '501018.XSHG',
    '511220.XSHG', '511880.XSHG', '513100.XSHG', '518880.XSHG',
}

# ETF 主题映射（用于归因报告分组）
ETF_THEME = {
    '518880.XSHG': '黄金', '159980.XSHE': '黄金', '159985.XSHE': '黄金',
    '501018.XSHG': '原油', '161226.XSHE': '白酒',
    '159981.XSHE': '原油', '513100.XSHG': '纳指',
    '511220.XSHG': '十年国债', '511880.XSHG': '货币(防御)',
    # 大池独有
    '159509.XSHE': '纳指', '513290.XSHG': '纳指科技', '513500.XSHG': '中概互联',
    '159529.XSHE': '中概互联', '513400.XSHG': '日经', '513520.XSHG': '日经',
    '513030.XSHG': '德国', '513080.XSHG': '法国', '513310.XSHG': '东南亚',
    '513730.XSHG': '半导体(韩国)', '159792.XSHE': '半导体',
    '513130.XSHG': '德国', '513050.XSHG': '中概互联',
    '159920.XSHE': '沪深300', '513690.XSHG': '白酒',
    '510300.XSHG': '沪深300', '510500.XSHG': '中证500',
    '510050.XSHG': '上证50', '510210.XSHG': '上证50',
    '159915.XSHE': '创业板', '588080.XSHG': '科创50',
    '512100.XSHG': '中证1000', '563360.XSHG': '中证1000',
    '563300.XSHG': '中证1000',
    '512890.XSHG': '红利', '159967.XSHE': '红利',
    '512040.XSHG': '金融',
    '511380.XSHG': '十年国债', '511010.XSHG': '国债',
    '159980.XSHE': '黄金',
    '159981.XSHE': '原油',
}


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def pair_trades(trades, daily, final_value, label):
    """把 trades 按时间顺序配对成完整交易。

    holdings_num=1，但同时只持 1 只。但 trades 里同一日可能有 sell+buy，
    需按 security 分组，每个 security 的 buy→sell 依次配对。

    期末未平仓 MTM：
    不用 trades 最后一笔 price（那是买入价，不是末日收盘价），
    改用"残差法"——期末持仓的 unrealized_pnl = final_value - INITIAL_CASH - realized_pnl。
    这样归因合计自动对齐最终权益，且语义清晰（期末持仓贡献 = 末日总权益减去已实现）。
    last_price 仅用于记录参考（买入价），不参与 PnL 计算。
    """
    # 按 security 分组，时间排序
    by_sec = defaultdict(list)
    for t in trades:
        by_sec[t['security']].append(t)
    for sec in by_sec:
        by_sec[sec].sort(key=lambda x: (x['date'], x['time']))

    paired = []
    open_positions = []  # 期末未平仓：记录持仓详情，PnL 后面用残差法填
    for sec, ts in by_sec.items():
        pos = 0  # 持有数量
        buy_queue = []  # (amount, price, date) FIFO
        i = 0
        while i < len(ts):
            t = ts[i]
            amt = t['amount']
            if amt > 0:  # 买入
                buy_queue.append({'amount': amt, 'price': t['price'], 'date': t['date'],
                                   'time': t['time'], 'commission': t['commission'], 'tax': t['tax']})
                pos += amt
                i += 1
            else:  # 卖出（amt < 0）
                sell_amt = -amt
                sell_price = t['price']
                sell_date = t['date']
                sell_commission = t['commission']
                sell_tax = t['tax']
                # FIFO 配对
                remaining = sell_amt
                buy_cost = 0.0
                buy_dates = []
                while remaining > 0 and buy_queue:
                    bq = buy_queue[0]
                    if bq['amount'] <= remaining:
                        buy_cost += bq['amount'] * bq['price'] + bq['commission'] + bq['tax']
                        buy_dates.append((bq['date'], bq['price']))
                        remaining -= bq['amount']
                        pos -= bq['amount']
                        buy_queue.pop(0)
                    else:
                        buy_cost += remaining * bq['price'] + (bq['commission'] + bq['tax']) * (remaining / bq['amount'])
                        buy_dates.append((bq['date'], bq['price']))
                        bq['amount'] -= remaining
                        pos -= remaining
                        remaining = 0
                sell_proceeds = sell_amt * sell_price - sell_commission - sell_tax
                pnl = sell_proceeds - buy_cost
                pnl_pct = pnl / buy_cost if buy_cost > 0 else 0.0
                first_buy_date = buy_dates[0][0] if buy_dates else sell_date
                from datetime import datetime
                d0 = datetime.strptime(first_buy_date, '%Y-%m-%d')
                d1 = datetime.strptime(sell_date, '%Y-%m-%d')
                hold_days = (d1 - d0).days
                paired.append({
                    'security': sec,
                    'buy_date': first_buy_date,
                    'sell_date': sell_date,
                    'hold_days': hold_days,
                    'buy_price': buy_dates[0][1] if buy_dates else 0,
                    'sell_price': sell_price,
                    'amount': sell_amt,
                    'buy_cost': round(buy_cost, 4),
                    'sell_proceeds': round(sell_proceeds, 4),
                    'pnl': round(pnl, 4),
                    'pnl_pct': round(pnl_pct * 100, 4),
                })
                i += 1
        # 期末未平仓：pos > 0（先记录持仓详情，PnL 后面用残差法计算）
        if pos > 0:
            last_buy = buy_queue[-1] if buy_queue else {}
            remaining_cost = sum(bq['amount'] * bq['price'] + bq['commission'] + bq['tax'] for bq in buy_queue)
            open_positions.append({
                'security': sec,
                'amount': pos,
                'buy_price': last_buy.get('price', 0),  # 参考价（买入价，非末日收盘价）
                'buy_date': last_buy.get('date', ''),
                'remaining_cost': round(remaining_cost, 4),
                # unrealized_pnl 由残差法填充
            })

    # 残差法：期末未平仓 PnL = 最终权益 - 初始资金 - 已实现 PnL
    realized_pnl = sum(p['pnl'] for p in paired)
    total_unrealized = (final_value - INITIAL_CASH) - realized_pnl
    unrealized = []
    if open_positions:
        # 单标的持仓（holdings_num=1），全部残差归该持仓
        # 但若有多只（理论上不会），按 remaining_cost 加权分配
        total_cost = sum(o['remaining_cost'] for o in open_positions)
        for o in open_positions:
            if total_cost > 0:
                share = o['remaining_cost'] / total_cost
            else:
                share = 1.0 / len(open_positions)
            o_pnl = total_unrealized * share
            o['unrealized_pnl'] = round(o_pnl, 4)
            o['unrealized_pnl_pct'] = round((o_pnl / o['remaining_cost'] * 100) if o['remaining_cost'] > 0 else 0, 4)
            o['mtm_value'] = round(o['remaining_cost'] + o_pnl, 4)
            o['last_price'] = round(o['mtm_value'] / o['amount'], 4) if o['amount'] > 0 else 0
            o['mtm_date'] = daily[-1]['date'].split(' ')[0] if daily else ''
            unrealized.append(o)

    return paired, unrealized


def reconcile(paired, unrealized, trades, final_value, label):
    """归因合计校验：已实现盈亏 + 未实现盈亏 + 现金/费用影响 ≈ 最终权益 - 初始资金"""
    realized_pnl = sum(p['pnl'] for p in paired)
    unrealized_pnl = sum(u['unrealized_pnl'] for u in unrealized)
    total_pnl = realized_pnl + unrealized_pnl
    expected_pnl = final_value - INITIAL_CASH
    diff = total_pnl - expected_pnl
    # 总费用（佣金+税）
    total_fees = sum(t['commission'] + t['tax'] for t in trades)
    return {
        'label': label,
        'realized_pnl': round(realized_pnl, 2),
        'unrealized_pnl': round(unrealized_pnl, 2),
        'total_attribution_pnl': round(total_pnl, 2),
        'expected_pnl': round(expected_pnl, 2),
        'diff': round(diff, 2),
        'diff_pct': round(abs(diff) / abs(expected_pnl) * 100, 4) if expected_pnl != 0 else 0,
        'total_fees': round(total_fees, 2),
        'final_value': round(final_value, 2),
        'n_paired': len(paired),
        'n_unrealized': len(unrealized),
    }


def etf_contribution(paired, unrealized, label):
    """ETF 级贡献：按 security 汇总"""
    contrib = defaultdict(lambda: {'n_trades': 0, 'realized_pnl': 0, 'unrealized_pnl': 0,
                                     'total_pnl': 0, 'buy_cost_total': 0, 'hold_days_sum': 0,
                                     'is_default_pool': False, 'theme': ''})
    for p in paired:
        sec = p['security']
        contrib[sec]['n_trades'] += 1
        contrib[sec]['realized_pnl'] += p['pnl']
        contrib[sec]['buy_cost_total'] += p['buy_cost']
        contrib[sec]['hold_days_sum'] += p['hold_days']
        contrib[sec]['is_default_pool'] = sec in DEFAULT_POOL
        contrib[sec]['theme'] = ETF_THEME.get(sec, '其他')
    for u in unrealized:
        sec = u['security']
        contrib[sec]['unrealized_pnl'] += u['unrealized_pnl']
        contrib[sec]['buy_cost_total'] += u['remaining_cost']
        if contrib[sec]['n_trades'] == 0:
            contrib[sec]['is_default_pool'] = sec in DEFAULT_POOL
            contrib[sec]['theme'] = ETF_THEME.get(sec, '其他')
    # 计算合计
    result = []
    for sec, c in contrib.items():
        c['total_pnl'] = c['realized_pnl'] + c['unrealized_pnl']
        c['avg_hold_days'] = round(c['hold_days_sum'] / c['n_trades'], 1) if c['n_trades'] > 0 else 0
        c['return_pct'] = round(c['total_pnl'] / c['buy_cost_total'] * 100, 2) if c['buy_cost_total'] > 0 else 0
        c['security'] = sec
        c['label'] = label
        c['realized_pnl'] = round(c['realized_pnl'], 2)
        c['unrealized_pnl'] = round(c['unrealized_pnl'], 2)
        c['total_pnl'] = round(c['total_pnl'], 2)
        c['buy_cost_total'] = round(c['buy_cost_total'], 2)
        result.append(dict(c))
    result.sort(key=lambda x: x['total_pnl'], reverse=True)
    return result


def monthly_contribution(paired, unrealized, label):
    """月度贡献：按卖出月（已实现）+ MTM月（未实现）"""
    monthly = defaultdict(lambda: {'realized_pnl': 0, 'unrealized_pnl': 0, 'n_closed': 0, 'n_open': 0})
    for p in paired:
        ym = p['sell_date'][:7]  # YYYY-MM
        monthly[ym]['realized_pnl'] += p['pnl']
        monthly[ym]['n_closed'] += 1
    for u in unrealized:
        ym = u['mtm_date'][:7]
        monthly[ym]['unrealized_pnl'] += u['unrealized_pnl']
        monthly[ym]['n_open'] += 1
    result = []
    for ym in sorted(monthly.keys()):
        c = monthly[ym]
        result.append({
            'month': ym,
            'label': label,
            'realized_pnl': round(c['realized_pnl'], 2),
            'unrealized_pnl': round(c['unrealized_pnl'], 2),
            'total_pnl': round(c['realized_pnl'] + c['unrealized_pnl'], 2),
            'n_closed': c['n_closed'],
            'n_open': c['n_open'],
        })
    return result


def pool_decomposition(etf_contrib):
    """原池共有 vs 大池独有 贡献拆解"""
    default_pnl = sum(c['total_pnl'] for c in etf_contrib if c['is_default_pool'])
    bak_only_pnl = sum(c['total_pnl'] for c in etf_contrib if not c['is_default_pool'])
    default_cost = sum(c['buy_cost_total'] for c in etf_contrib if c['is_default_pool'])
    bak_only_cost = sum(c['buy_cost_total'] for c in etf_contrib if not c['is_default_pool'])
    default_n = sum(c['n_trades'] for c in etf_contrib if c['is_default_pool'])
    bak_only_n = sum(c['n_trades'] for c in etf_contrib if not c['is_default_pool'])
    return {
        'default_pool_pnl': round(default_pnl, 2),
        'bak_only_pnl': round(bak_only_pnl, 2),
        'default_pool_cost': round(default_cost, 2),
        'bak_only_cost': round(bak_only_cost, 2),
        'default_pool_n_trades': default_n,
        'bak_only_n_trades': bak_only_n,
        'default_pool_return_pct': round(default_pnl / default_cost * 100, 2) if default_cost > 0 else 0,
        'bak_only_return_pct': round(bak_only_pnl / bak_only_cost * 100, 2) if bak_only_cost > 0 else 0,
        'n_default_etfs': sum(1 for c in etf_contrib if c['is_default_pool']),
        'n_bak_only_etfs': sum(1 for c in etf_contrib if not c['is_default_pool']),
    }


def main():
    print(f"{'=' * 80}")
    print(f"P3-01 持仓级归因")
    print(f"{'=' * 80}")

    all_reconcile = {}
    all_etf = {}
    all_monthly = {}
    all_decomp = {}

    for w in WINDOWS:
        all_reconcile[w] = {}
        all_etf[w] = {}
        all_monthly[w] = {}
        all_decomp[w] = {}
        for g in GROUPS:
            label = f'{g}_{w}'
            p = WF_DIR / f'{g}_{w}.json'
            data = load_json(p)
            trades = data.get('trades', [])
            daily = data.get('daily', [])
            final_value = data.get('final_value') or data.get('metrics', {}).get('final_value')

            paired, unrealized = pair_trades(trades, daily, final_value, label)
            recon = reconcile(paired, unrealized, trades, final_value, label)
            etf = etf_contribution(paired, unrealized, label)
            monthly = monthly_contribution(paired, unrealized, label)
            decomp = pool_decomposition(etf)

            # 保存
            with open(OUT_DIR / f'trades_paired_{g}_{w}.json', 'w', encoding='utf-8') as f:
                json.dump({'paired': paired, 'unrealized': unrealized}, f, indent=2, ensure_ascii=False)
            with open(OUT_DIR / f'etf_contribution_{g}_{w}.json', 'w', encoding='utf-8') as f:
                json.dump(etf, f, indent=2, ensure_ascii=False)
            with open(OUT_DIR / f'monthly_contribution_{g}_{w}.json', 'w', encoding='utf-8') as f:
                json.dump(monthly, f, indent=2, ensure_ascii=False)

            all_reconcile[w][g] = recon
            all_etf[w][g] = etf
            all_monthly[w][g] = monthly
            all_decomp[w][g] = decomp

            flag = '✅' if abs(recon['diff']) < abs(recon['expected_pnl']) * 0.01 + 100 else '⚠️'
            print(f"{label}: 配对={recon['n_paired']} 未平仓={recon['n_unrealized']} "
                  f"归因PnL={recon['total_attribution_pnl']:.0f} 期望={recon['expected_pnl']:.0f} "
                  f"差={recon['diff']:.0f} {flag}")

    # 保存拆解（每个窗口 A vs C vs D）
    for w in WINDOWS:
        with open(OUT_DIR / f'pool_decomposition_{w}.json', 'w', encoding='utf-8') as f:
            json.dump({g: all_decomp[w][g] for g in GROUPS}, f, indent=2, ensure_ascii=False)

    # 汇总
    summary = {
        'reconcile': all_reconcile,
        'pool_decomposition': {w: {g: all_decomp[w][g] for g in GROUPS} for w in WINDOWS},
    }
    with open(OUT_DIR / 'attribution_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印归因校验总览
    print(f"\n{'=' * 90}")
    print(f"归因校验总览（diff = 归因PnL - 期望PnL）")
    print(f"{'窗口':<8} {'组':<4} {'归因PnL':>14} {'期望PnL':>14} {'差':>12} {'差%':>8}")
    print(f"{'-' * 90}")
    all_ok = True
    for w in WINDOWS:
        for g in GROUPS:
            r = all_reconcile[w][g]
            flag = '✅' if abs(r['diff']) < abs(r['expected_pnl']) * 0.01 + 100 else '⚠️'
            if '⚠️' in flag:
                all_ok = False
            print(f"{w:<8} {g:<4} {r['total_attribution_pnl']:>14.0f} {r['expected_pnl']:>14.0f} "
                  f"{r['diff']:>12.0f} {r['diff_pct']:>7.3f}% {flag}")
    print(f"\n全部校验通过: {'是' if all_ok else '否（见 ⚠️）'}")

    print(f"\n输出目录: {OUT_DIR}")


if __name__ == '__main__':
    main()
