"""P0 任务三：Calmar 计算单元测试

验证 Calmar = 年化收益率 / 最大回撤（不是总收益 / 最大回撤）
对多年度区间差异显著：两年期总收益100%时，年化≈41.4%，Calmar 差约 2 倍。
"""
import sys
import os
import numpy as np

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'qixing_optimize', 'scripts'))

from run_sweep import compute_metrics


def _make_res(values, initial_cash=1000000):
    """构造 compute_metrics 需要的 res 字典"""
    daily = [{'value': v} for v in values]
    trades = []
    return {'daily': daily, 'trades': trades, 'final_value': values[-1]}


def test_calmar_one_year():
    """一年样例：252 个交易日，总收益 50%，最大回撤 10%
    一年期年化收益 = 总收益 = 50%
    Calmar = 0.5 / 0.1 = 5.0
    """
    # 构造净值序列：初始 100 万，先涨到 150 万，中间回撤到 135 万（10% 回撤），最终 150 万
    n_days = 252
    values = np.zeros(n_days)
    # 前 126 天：100 万 → 150 万（线性增长）
    values[:126] = np.linspace(1000000, 1500000, 126)
    # 第 126-189 天：150 万 → 135 万（10% 回撤）
    values[126:189] = np.linspace(1500000, 1350000, 63)
    # 第 189-252 天：135 万 → 150 万（恢复）
    values[189:] = np.linspace(1350000, 1500000, n_days - 189)

    res = _make_res(values.tolist())
    m = compute_metrics(res, 'test_1y', 0.0)

    print(f"一年样例: 总收益={m['total_return_pct']}%, 年化={m['annualized_return_pct']}%, "
          f"回撤={m['max_drawdown_pct']}%, Calmar={m['calmar']}, 天数={m['n_trade_days']}")

    # 一年期：年化收益 = 总收益
    assert m['n_trade_days'] == 252, f"天数应为252，实际{m['n_trade_days']}"
    assert abs(m['total_return_pct'] - 50.0) < 0.1, f"总收益应≈50%，实际{m['total_return_pct']}"
    assert abs(m['annualized_return_pct'] - 50.0) < 0.1, f"一年期年化应=总收益≈50%，实际{m['annualized_return_pct']}"
    assert abs(m['max_drawdown_pct'] - 10.0) < 0.5, f"回撤应≈10%，实际{m['max_drawdown_pct']}"
    # Calmar = 0.5 / 0.1 = 5.0
    assert abs(m['calmar'] - 5.0) < 0.3, f"Calmar应≈5.0，实际{m['calmar']}"
    print("✅ test_calmar_one_year 通过")


def test_calmar_two_years():
    """两年样例：504 个交易日，总收益 100%，最大回撤 20%
    年化收益 = (1+1.0)^(252/504) - 1 = 2^0.5 - 1 ≈ 41.42%
    Calmar = 0.4142 / 0.2 ≈ 2.071
    旧公式（错误）= 1.0 / 0.2 = 5.0（高估约 2.4 倍）
    """
    n_days = 504
    values = np.zeros(n_days)
    # 前 252 天：100 万 → 200 万（涨 100%）
    values[:252] = np.linspace(1000000, 2000000, 252)
    # 第 252-378 天：200 万 → 160 万（20% 回撤）
    values[252:378] = np.linspace(2000000, 1600000, 126)
    # 第 378-504 天：160 万 → 200 万（恢复到 200 万，总收益 100%）
    values[378:] = np.linspace(1600000, 2000000, n_days - 378)

    res = _make_res(values.tolist())
    m = compute_metrics(res, 'test_2y', 0.0)

    print(f"两年样例: 总收益={m['total_return_pct']}%, 年化={m['annualized_return_pct']}%, "
          f"回撤={m['max_drawdown_pct']}%, Calmar={m['calmar']}, 天数={m['n_trade_days']}")

    assert m['n_trade_days'] == 504, f"天数应为504，实际{m['n_trade_days']}"
    assert abs(m['total_return_pct'] - 100.0) < 0.1, f"总收益应≈100%，实际{m['total_return_pct']}"
    # 年化 = 2^0.5 - 1 ≈ 41.42%
    expected_ann = (2.0 ** 0.5 - 1) * 100
    assert abs(m['annualized_return_pct'] - expected_ann) < 0.5, \
        f"年化应≈{expected_ann:.2f}%，实际{m['annualized_return_pct']}"
    assert abs(m['max_drawdown_pct'] - 20.0) < 0.5, f"回撤应≈20%，实际{m['max_drawdown_pct']}"
    # Calmar = 0.4142 / 0.2 ≈ 2.071
    expected_calmar = (2.0 ** 0.5 - 1) / 0.2
    assert abs(m['calmar'] - expected_calmar) < 0.1, \
        f"Calmar应≈{expected_calmar:.3f}，实际{m['calmar']}"
    # 验证旧公式确实高估：旧 Calmar = 1.0/0.2 = 5.0，新 Calmar ≈ 2.071
    assert m['calmar'] < 3.0, f"两年期 Calmar 应 < 3.0（年化口径），实际{m['calmar']}，旧公式会给出5.0"
    print("✅ test_calmar_two_years 通过")


def test_calmar_old_formula_would_differ():
    """验证旧公式（total_return/max_dd）与新公式（annualized/max_dd）在两年期有明显差异"""
    n_days = 504
    values = np.zeros(n_days)
    values[:252] = np.linspace(1000000, 2000000, 252)
    values[252:378] = np.linspace(2000000, 1600000, 126)
    values[378:] = np.linspace(1600000, 2000000, n_days - 378)

    res = _make_res(values.tolist())
    m = compute_metrics(res, 'test_diff', 0.0)

    total_return = m['total_return_pct'] / 100
    annualized = m['annualized_return_pct'] / 100
    max_dd = m['max_drawdown_pct'] / 100

    old_calmar = total_return / max_dd
    new_calmar = m['calmar']

    print(f"旧公式 Calmar = {old_calmar:.3f}（total_return/max_dd）")
    print(f"新公式 Calmar = {new_calmar:.3f}（annualized/max_dd）")
    print(f"差异倍数 = {old_calmar / new_calmar:.2f}x")

    assert old_calmar > new_calmar * 1.5, "两年期旧公式应明显高估 Calmar"
    assert abs(new_calmar - annualized / max_dd) < 0.01, "新公式应等于 annualized/max_dd"
    print("✅ test_calmar_old_formula_would_differ 通过")


if __name__ == '__main__':
    test_calmar_one_year()
    test_calmar_two_years()
    test_calmar_old_formula_would_differ()
    print("\n🎉 所有 Calmar 测试通过")
