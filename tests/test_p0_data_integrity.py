"""P0 数据完整性修复 — 自动化测试

覆盖用户要求的 8 项测试：
1. parity 模式遇到 NAV 缺失必须报错
2. realistic 模式遇到 NAV 缺失必须排除候选
3. legacy_invalid 模式能复现 0 溢价，但结果带 invalid 标记
4. NAV 预检能发现池内未进入 Top3 的缺失 ETF
5. Calmar 一年样例
6. Calmar 两年样例
7. 2024 原 7 只池 parity 短区间结果不得因本次修复发生变化（基准值 981257.41）
8. 当天 NAV 存在、前一交易日 NAV 缺失时，parity 必须停止

注意：测试 1-4、7-8 需要完整引擎环境，运行时间较长（每条约 30-60 秒）。
测试 5-6 为纯计算测试，瞬间完成。

运行方式：
  python tests/test_p0_data_integrity.py          # 全部测试
  python tests/test_p0_data_integrity.py --quick   # 仅快速测试（Calmar）
"""
import sys
import os
import json
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'qixing_optimize', 'scripts'))

# 大池 37 只（POOL_BAK，去掉缺失的 159201）
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

# 短区间参数（3 个交易日，快速验证）
SHORT_START = '2024-01-02'
SHORT_END = '2024-01-05'


# ========== 测试 1-4：NAV 模式测试（需要引擎环境）==========

def test_1_parity_nav_missing_aborts():
    """测试1：parity 模式遇到 NAV 缺失必须报错"""
    from run_local import QixingParityRunner
    with pytest.raises(RuntimeError, match="NAV_PRECHECK|PARITY"):
        QixingParityRunner(
            SHORT_START, SHORT_END,
            param_overrides={'etf_pool': POOL_BAK},
            nav_mode='parity'
        ).run()


def test_2_realistic_nav_missing_excludes():
    """测试2：realistic 模式遇到 NAV 缺失必须排除候选（返回 None 让策略跳过）"""
    from run_local import QixingParityRunner
    res = QixingParityRunner(
        SHORT_START, SHORT_END,
        param_overrides={'etf_pool': POOL_BAK},
        nav_mode='realistic'
    ).run()
    # realistic 模式不应标记 invalid
    assert res['invalid_due_to_nav_fail_open'] is False, "realistic 模式不应标记 invalid"
    assert res['nav_mode'] == 'realistic'
    # 预检应识别 28 只无 NAV ETF
    precheck = res.get('nav_precheck', {})
    assert precheck.get('missing_count', 0) == 28, \
        f"预检应识别 28 只无 NAV ETF，实际 {precheck.get('missing_count')}"


def test_3_legacy_invalid_marks_results():
    """测试3：legacy_invalid 模式能复现 0 溢价，但结果带 invalid 标记"""
    from run_local import QixingParityRunner
    res = QixingParityRunner(
        SHORT_START, SHORT_END,
        param_overrides={'etf_pool': POOL_BAK},
        nav_mode='legacy_invalid'
    ).run()
    assert res['invalid_due_to_nav_fail_open'] is True, "legacy_invalid 模式必须标记 invalid"
    assert res['nav_mode'] == 'legacy_invalid'


def test_4_precheck_finds_non_top3_missing():
    """测试4：NAV 预检能发现池内未进入 Top3 的缺失 ETF

    预检不是只检查实际进入溢价检查的候选，而是检查整个池的 NAV 覆盖。
    即使 Top3 候选恰好有 NAV，预检也应报告池内 28 只无 NAV 的 ETF。
    """
    from run_local import QixingParityRunner
    # 用 realistic 模式运行（预检不会终止）
    res = QixingParityRunner(
        SHORT_START, SHORT_END,
        param_overrides={'etf_pool': POOL_BAK},
        nav_mode='realistic'
    ).run()
    precheck = res.get('nav_precheck', {})
    # 预检必须报告 28 只缺失 ETF，即使运行时日志可能显示"无缺失"
    assert precheck.get('missing_count', 0) == 28, \
        f"预检应发现 28 只无 NAV ETF（不依赖运行时是否触发），实际 {precheck.get('missing_count')}"
    # 运行时日志（_nav_missing_log）可能为空（因为 Top3 有 NAV），但预检必须完整
    nav_report = res.get('nav_report', {})
    # nav_report 的 total_missing 是运行时统计，可能为 0
    # 但 precheck 的 missing_count 是预检统计，必须为 28
    assert precheck['missing_count'] >= nav_report.get('total_missing', 0), \
        "预检缺失数应 >= 运行时缺失数"


# ========== 测试 5-6：Calmar 计算测试（纯计算，快速）==========

def test_5_calmar_one_year():
    """测试5：Calmar 一年样例"""
    from run_sweep import compute_metrics
    import numpy as np
    n_days = 252
    values = np.zeros(n_days)
    values[:126] = np.linspace(1000000, 1500000, 126)
    values[126:189] = np.linspace(1500000, 1350000, 63)
    values[189:] = np.linspace(1350000, 1500000, n_days - 189)
    res = {'daily': [{'value': v} for v in values], 'trades': [], 'final_value': values[-1]}
    m = compute_metrics(res, 'test_1y', 0.0)
    assert m['n_trade_days'] == 252
    assert abs(m['total_return_pct'] - 50.0) < 0.1
    assert abs(m['annualized_return_pct'] - 50.0) < 0.1  # 一年期年化=总收益
    assert abs(m['calmar'] - 5.0) < 0.3  # 0.5/0.1=5.0


def test_6_calmar_two_years():
    """测试6：Calmar 两年样例（验证年化口径，非总收益口径）"""
    from run_sweep import compute_metrics
    import numpy as np
    n_days = 504
    values = np.zeros(n_days)
    values[:252] = np.linspace(1000000, 2000000, 252)
    values[252:378] = np.linspace(2000000, 1600000, 126)
    values[378:] = np.linspace(1600000, 2000000, n_days - 378)
    res = {'daily': [{'value': v} for v in values], 'trades': [], 'final_value': values[-1]}
    m = compute_metrics(res, 'test_2y', 0.0)
    assert m['n_trade_days'] == 504
    assert abs(m['total_return_pct'] - 100.0) < 0.1
    expected_ann = (2.0 ** 0.5 - 1) * 100  # ≈ 41.42%
    assert abs(m['annualized_return_pct'] - expected_ann) < 0.5
    # 旧公式 Calmar = 1.0/0.2 = 5.0（错误），新公式 ≈ 2.071
    assert m['calmar'] < 3.0, f"两年期 Calmar 应 < 3.0（年化口径），实际 {m['calmar']}"


# ========== 测试 7：对齐结果不变性测试 ==========

def test_7_original_pool_parity_unchanged():
    """测试7：2024 原 7 只池 parity 短区间结果不得因本次修复发生变化

    原 7 只池全部有 NAV 数据，parity 模式应正常通过，
    且交易结果应与修复前一致（最终权益 981257.41）。
    """
    from run_local import QixingParityRunner
    res = QixingParityRunner(
        SHORT_START, SHORT_END,
        nav_mode='parity'  # 默认模式，原池有完整 NAV
    ).run()
    # 原 7 只池有完整 NAV，parity 模式应正常通过
    assert res['invalid_due_to_nav_fail_open'] is False
    assert res['nav_mode'] == 'parity'
    # 预检应无缺失
    precheck = res.get('nav_precheck', {})
    assert precheck.get('missing_count', 0) == 0, \
        f"原 7 只池应无 NAV 缺失，实际 {precheck.get('missing_count')}"
    # 运行时也应无缺失
    nav_report = res.get('nav_report', {})
    assert nav_report.get('total_missing', 0) == 0, "原 7 只池运行时不应有 NAV 缺失"
    # 最终权益应与修复前基准值一致（允许微小浮点差异 0.01 元）
    # 基准值 981257.41 来自修复前多次运行的稳定结果
    assert abs(res['final_value'] - 981257.41) < 0.02, \
        f"最终权益应与修复前基准 981257.41 一致，实际 {res['final_value']}"


# ========== 测试 8：前一交易日 NAV 边界测试 ==========

def test_8_prev_day_nav_missing_aborts():
    """测试8：当天 NAV 存在、前一交易日 NAV 缺失时，parity 必须停止

    策略 get_premium_rate(code, prev_date) 使用前一交易日的 NAV。
    NAV 文件从 2023-10-09 开始，若回测从 2023-10-09 开始：
    - 交易日 T0 = 2023-10-09，需要前一交易日 T-1（如 2023-10-08 或 09-28）
    - T-1 不在 NAV 文件中（文件从 10-09 开始）
    - 因此原 7 只池（有 NAV 数据）在 T0 的前一交易日也缺 NAV
    - parity 模式必须终止

    这验证预检检查的是 T-1 而非 T 当天。
    """
    from run_local import QixingParityRunner
    # 回测从 NAV 文件起始日开始，前一交易日必然在文件外
    with pytest.raises(RuntimeError, match="NAV_PRECHECK|PARITY"):
        QixingParityRunner(
            '2023-10-09', '2023-10-10',
            nav_mode='parity'  # 默认原 7 只池
        ).run()


if __name__ == '__main__':
    # 直接运行模式：支持 --quick 参数跳过引擎测试
    if '--quick' in sys.argv:
        test_5_calmar_one_year()
        test_6_calmar_two_years()
        print("✅ 快速测试通过（Calmar 一年/两年）")
    else:
        # 运行所有测试
        test_funcs = [
            test_1_parity_nav_missing_aborts,
            test_2_realistic_nav_missing_excludes,
            test_3_legacy_invalid_marks_results,
            test_4_precheck_finds_non_top3_missing,
            test_5_calmar_one_year,
            test_6_calmar_two_years,
            test_7_original_pool_parity_unchanged,
            test_8_prev_day_nav_missing_aborts,
        ]
        for i, tf in enumerate(test_funcs, 1):
            print(f"\n{'=' * 60}")
            print(f"运行测试 {i}: {tf.__name__}")
            print(f"{'=' * 60}")
            tf()
            print(f"✅ 测试 {i} 通过")
        print(f"\n🎉 所有 {len(test_funcs)} 项测试通过")
