"""
聚宽净值导出脚本 (独立，不修改权威策略)
=======================================
在聚宽研究环境中运行此脚本，导出七星高照所需的基金净值数据。

使用方式:
  1. 在聚宽研究平台创建新的Notebook
  2. 复制此脚本内容并运行
  3. 下载生成的 CSV 文件
  4. 放到 D:\\Work Space\\HData\\imports\\qixing_fund_nav\\

注意: 此脚本独立于权威策略，不依赖或修改七星高照 V1.7 母版。
"""

import pandas as pd

# ============ 基金代码列表 ============
# (在聚宽环境中代码格式为 XSHG/XSHE)
SECURITIES = [
    '510050.XSHG',   # 50ETF
    '513100.XSHG',   # 纳指ETF
    '159920.XSHE',   # 恒生ETF
    '518880.XSHG',   # 黄金ETF
    '511220.XSHG',   # 城投债ETF
    '501018.XSHG',   # 南方原油LOF
    '161226.XSHE',   # 白银LOF
    '511880.XSHG',   # 银华日利(防御)
]

START_DATE = '2023-10-01'
END_DATE   = '2024-12-31'

# ============ 导出 unit_net_value ============
print("正在导出 unit_net_value ...")
df_unit = get_extras(
    'unit_net_value',
    SECURITIES,
    start_date=START_DATE,
    end_date=END_DATE,
    df=True
)

print(f"unit_net_value 原始形状: {df_unit.shape}")
print(f"列名: {list(df_unit.columns)}")
print(f"行索引范围: {df_unit.index[0]} ~ {df_unit.index[-1]}")

# 转为长表: nav_date, code, unit_net_value
rows = []
for code in SECURITIES:
    if code in df_unit.columns:
        series = df_unit[code]
        for dt_idx, val in series.items():
            if pd.notna(val):
                rows.append({'nav_date': str(dt_idx.date()), 'code': code, 'unit_net_value': float(val)})
            else:
                # 空值也保留
                rows.append({'nav_date': str(dt_idx.date()), 'code': code, 'unit_net_value': ''})

df_unit_long = pd.DataFrame(rows)
df_unit_long = df_unit_long.sort_values(['nav_date', 'code']).reset_index(drop=True)
print(f"unit_net_value 长表行数: {len(df_unit_long)}")
print(f"有效值行数: {df_unit_long['unit_net_value'].apply(lambda x: x != '').sum()}")
print(f"每只基金记录数:\n{df_unit_long.groupby('code').size()}")
# Save: use -- to pass path to log
# In JQ, you can download via: df_unit_long.to_csv('qixing_unit_net_value.csv', index=False)
# Or use log.info() to print serialized content

print("\n--- CSV 头部预览 (unit_net_value) ---")
print(df_unit_long.head(20).to_string(index=False))


# ============ 导出 acc_net_value ============
print("\n\n正在导出 acc_net_value ...")
try:
    df_acc = get_extras(
        'acc_net_value',
        SECURITIES,
        start_date=START_DATE,
        end_date=END_DATE,
        df=True
    )
    print(f"acc_net_value 原始形状: {df_acc.shape}")

    rows_acc = []
    for code in SECURITIES:
        if code in df_acc.columns:
            series = df_acc[code]
            for dt_idx, val in series.items():
                if pd.notna(val):
                    rows_acc.append({'nav_date': str(dt_idx.date()), 'code': code, 'acc_net_value': float(val)})
                else:
                    rows_acc.append({'nav_date': str(dt_idx.date()), 'code': code, 'acc_net_value': ''})

    df_acc_long = pd.DataFrame(rows_acc)
    df_acc_long = df_acc_long.sort_values(['nav_date', 'code']).reset_index(drop=True)
    print(f"acc_net_value 长表行数: {len(df_acc_long)}")
    print(f"有效值行数: {df_acc_long['acc_net_value'].apply(lambda x: x != '').sum()}")
    print(f"每只基金记录数:\n{df_acc_long.groupby('code').size()}")
    print("\n--- CSV 头部预览 (acc_net_value) ---")
    print(df_acc_long.head(20).to_string(index=False))
except Exception as e:
    print(f"acc_net_value 导出失败: {e}")
    df_acc_long = pd.DataFrame()

# ============ 保存文件 ============
# 在聚宽研究环境中，以下方式将 DataFrame 保存为可下载的 CSV:
print("\n" + "=" * 60)
print("请在聚宽界面使用以下方式下载文件:")
print("=" * 60)
print("1. 左侧文件管理器中会生成以下文件:")
print("   - qixing_unit_net_value_20231001_20241231.csv")
print("   - qixing_acc_net_value_20231001_20241231.csv")
print("2. 右键点击文件 → 下载")
print("3. 将下载的文件放入:")
print("   D:\\Work Space\\HData\\imports\\qixing_fund_nav\\")
print("=" * 60)

# 实际写入文件
df_unit_long.to_csv('qixing_unit_net_value_20231001_20241231.csv', index=False, encoding='utf-8-sig')
if len(df_acc_long) > 0:
    df_acc_long.to_csv('qixing_acc_net_value_20231001_20241231.csv', index=False, encoding='utf-8-sig')
    print("✅ 两个 CSV 文件已生成")
else:
    print("✅ unit_net_value CSV 已生成 (acc_net_value 导出失败，仅生成一个文件)")

# ============ 验证 513100.XSHG 2023-12-29 ============
print("\n\n============ 关键验证: 513100.XSHG 净值 ============")
test = df_unit_long[(df_unit_long['code'] == '513100.XSHG') & (df_unit_long['nav_date'] == '2023-12-29')]
if len(test) > 0:
    val = test.iloc[0]['unit_net_value']
    print(f"513100.XSHG nav_date=2023-12-29 unit_net_value={val}")
    print(f"期望值: 1.203 (来自聚宽诊断日志,溢价率0.75%=(1.212-1.203)/1.203)")
    try:
        if abs(float(val) - 1.203) < 0.01:
            print("✅ 验证通过! 净值与聚宽诊断一致")
        else:
            print(f"⚠️ 偏差较大: 差值={abs(float(val)-1.203):.4f}")
    except:
        print("⚠️ 无法比较 (非数值)")
else:
    print("❌ 未找到 513100.XSHG 的 2023-12-29 净值记录!")
    print("   请检查: start_date 是否覆盖到该日期, 或聚宽是否有该日数据")
