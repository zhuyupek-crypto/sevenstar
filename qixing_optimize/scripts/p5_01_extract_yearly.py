"""P5-01 基线年度绩效提取 + 与 P5-00 本地 O7_baseline 2024 年对比"""
import json
from pathlib import Path

BASELINE = Path(r'd:\Work Space\他山之石\七星高照ETF轮动\qixing_optimize\runs\p5_01_localization_baseline\localization_baseline_2019-12-06_2026-06-12.json')
O7_2024 = Path(r'd:\Work Space\他山之石\七星高照ETF轮动\qixing_optimize\runs\p4_01_core_pool_validation\O7_baseline_2024.json')

with open(BASELINE, 'r', encoding='utf-8') as f:
    bl = json.load(f)
with open(O7_2024, 'r', encoding='utf-8') as f:
    o7 = json.load(f)

daily = bl['daily']
trades = bl['trades']

# 按年分组
from collections import defaultdict
yearly_nav = defaultdict(list)
for d in daily:
    date_str = d['date'].split(' ')[0] if ' ' in d['date'] else d['date']
    year = date_str[:4]
    yearly_nav[year].append((date_str, d['value']))

print("=" * 80)
print("P5-01 本地化母版基线 年度绩效（2019-12-06 ~ 2026-06-12）")
print("=" * 80)
print(f"{'年份':<8} {'首日':<12} {'末日':<12} {'年初净值':>14} {'年末净值':>14} {'年度收益%':>10} {'交易日':>6}")
print("-" * 80)

prev_year_end = None
for year in sorted(yearly_nav.keys()):
    days = yearly_nav[year]
    first_date, first_nav = days[0]
    last_date, last_nav = days[-1]

    # 年初净值：如果是第一年，用初始资金 100 万；否则用上年末净值
    if prev_year_end is None:
        start_nav = 1000000.0
    else:
        start_nav = prev_year_end

    year_ret = (last_nav / start_nav - 1) * 100
    print(f"{year:<8} {first_date:<12} {last_date:<12} {start_nav:>14,.2f} {last_nav:>14,.2f} {year_ret:>10.2f} {len(days):>6}")
    prev_year_end = last_nav

# 总绩效
print("-" * 80)
final = daily[-1]['value']
total_ret = (final / 1000000 - 1) * 100
print(f"{'全区间':<8} {daily[0]['date']:<12} {daily[-1]['date']:<12} {1000000:>14,.2f} {final:>14,.2f} {total_ret:>10.2f} {len(daily):>6}")

# 2024 年对比 P5-00 本地 O7_baseline
print("\n" + "=" * 80)
print("2024 年对比：P5-01 基线 vs P5-00 本地 O7_baseline")
print("=" * 80)

# P5-01 基线 2024 年
days_2024 = yearly_nav['2024']
bl_2024_start = days_2024[0][1]
bl_2024_end = days_2024[-1][1]
bl_2024_ret = (bl_2024_end / bl_2024_start - 1) * 100

# P5-00 本地 O7_baseline 2024 年
o7_daily = o7['daily']
o7_2024_start = o7_daily[0]['value']
o7_2024_end = o7_daily[-1]['value']
o7_2024_ret = (o7_2024_end / 1000000 - 1) * 100  # O7_baseline 2024 单独跑，初始 100 万

print(f"{'指标':<20} {'P5-01基线(连续)':>20} {'O7_baseline(单独)':>20} {'差异':>10}")
print("-" * 80)
print(f"{'2024年初净值':<20} {bl_2024_start:>20,.2f} {1000000:>20,.2f} {bl_2024_start-1000000:>10,.2f}")
print(f"{'2024年末净值':<20} {bl_2024_end:>20,.2f} {o7_2024_end:>20,.2f} {bl_2024_end-o7_2024_end:>10,.2f}")
print(f"{'2024年收益率%':<20} {bl_2024_ret:>20.2f} {o7_2024_ret:>20.2f} {bl_2024_ret-o7_2024_ret:>10.2f}")
print(f"{'交易日数':<20} {len(days_2024):>20} {len(o7_daily):>20} {len(days_2024)-len(o7_daily):>10}")

# 逐日净值对比 2024 年
print("\n逐日净值对比（2024 年，前 10 天 + 后 5 天）:")
print(f"{'日期':<12} {'P5-01基线':>14} {'O7_baseline':>14} {'差异':>14} {'差异%':>10}")
print("-" * 80)

o7_nav_by_date = {d['date'].split(' ')[0]: d['value'] for d in o7_daily}
count = 0
for date_str, bl_nav in days_2024:
    o7_nav = o7_nav_by_date.get(date_str)
    if o7_nav is not None:
        diff = bl_nav - o7_nav
        # P5-01 是连续净值，O7 是 100 万起始，需要归一化
        # 用百分比收益率对比
        bl_ret = (bl_nav / bl_2024_start - 1) * 100
        o7_ret = (o7_nav / 1000000 - 1) * 100
        diff_pct = bl_ret - o7_ret
        if count < 10 or count >= len(days_2024) - 5:
            print(f"{date_str:<12} {bl_nav:>14,.2f} {o7_nav:>14,.2f} {diff:>14,.2f} {diff_pct:>10.4f}")
        count += 1

print(f"\n共 {count} 个对齐日")
