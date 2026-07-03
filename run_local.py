"""
七星高照 V1.7 母版 本地对齐运行入口（性能优化版）
=====================================================
基于 local_quant/run_qixing_v17_parity.py，增加两项性能优化：
  1. 注册活跃证券集合（g.etf_pool + g.defensive_etf），
     使 _get_daily_*_snapshot 只查 8 只 ETF 而非全市场 5000+ 只股票。
  2. 覆盖 get_name 为快速版本（走 get_security_info 内存缓存），
     避免原版 get_name 调用 get_current_data()[sec].name 触发全市场快照后抛异常。

运行方式:
  python run_local.py 2024-01-01 2024-01-10     # 短区间诊断
  python run_local.py 2024-01-01 2024-12-31     # 全年回测
"""
import sys
import os
import time
import traceback
from pathlib import Path
import pandas as pd
import numpy  # noqa: F401 — 确保 np 可用

import sys as _sys


def _sync_namespace_to_jqdata(engine):
    """将 engine.namespace 的变更同步到 sys.modules['jqdata']。"""
    for mod_name in ['jqdata', 'jqdatasdk']:
        mock_mod = _sys.modules.get(mod_name)
        if mock_mod is not None:
            for k, v in engine.namespace.items():
                setattr(mock_mod, k, v)


# ---- 数据环境 ----
os.environ.setdefault('LOCAL_QUANT_HDATA_SOURCE', 'core')
os.environ.setdefault('HDATA_ROOT', r'D:\Work Space\HData')

# 将 local_quant 加入 sys.path
_LOCAL_QUANT = r'D:\Work Space\local_quant'
if _LOCAL_QUANT not in _sys.path:
    _sys.path.insert(0, _LOCAL_QUANT)

# ---- 加载策略源码 ----
STRATEGY_PATH = os.path.join(os.path.dirname(__file__), '七星高照ETF轮动策略v1.7-母版.py')
with open(STRATEGY_PATH, 'r', encoding='utf-8') as f:
    STRATEGY_CODE = f.read()

# ---- 引擎 ----
from engine import Engine


class QixingParityRunner:
    """运行器：加载策略、注入兼容层与性能优化、运行回测、输出诊断"""

    def __init__(self, start_date, end_date, param_overrides=None, score_mode='baseline',
                 nav_mode='parity'):
        """
        Args:
            nav_mode: NAV 缺失处理策略
                - 'parity':         NAV 缺失立即终止回测（默认，用于对齐聚宽）
                - 'realistic':      NAV 缺失返回 None，策略原生逻辑跳过该 ETF（用于研究/实盘）
                - 'legacy_invalid': 保留旧版 0 溢价兜底（仅用于复现旧污染结果，禁止用于新报告）
        """
        self.start_date = start_date
        self.end_date = end_date
        self.engine = None
        self.diag_records = []
        self._name_cache = {}  # get_name 缓存
        # 优化扫描用：覆盖策略 initialize() 的默认参数（None 时保持对齐基线行为）
        self.param_overrides = param_overrides or {}
        # 得分模式：baseline（原版）/ multi_period（25d+60d加权）/ vol_adjusted（波动率惩罚）
        self.score_mode = score_mode
        # NAV 缺失处理模式
        if nav_mode not in ('parity', 'realistic', 'legacy_invalid'):
            raise ValueError(f"nav_mode 必须为 parity/realistic/legacy_invalid， got {nav_mode}")
        self.nav_mode = nav_mode
        # NAV 缺失统计（记录每只 ETF 的净值缺失事件）
        self._nav_missing_log = []
        # legacy_invalid 模式标记：结果文件必须包含此标记
        self._invalid_due_to_nav_fail_open = (nav_mode == 'legacy_invalid')

    def run(self):
        engine = Engine(
            STRATEGY_CODE,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_cash=1000000,
            frequency='daily',
        )
        # 引擎的 namespace 在 set_active_securities() 内部创建，
        # 先用占位列表初始化 namespace，post_exec_hook 中再注册实际集合。
        engine.set_active_securities(['513100.XSHG'])  # 占位，后续覆盖
        engine.namespace['pd'] = pd
        engine.namespace['math'] = __import__('math')

        # ---- 七星高照净值适配器 ----
        sys.path.insert(0, _LOCAL_QUANT)
        from research.qixing_nav_adapter import QixingNavAdapter

        _nav_file = (Path(os.environ.get('HDATA_ROOT', r'D:\Work Space\HData'))
                     / 'data' / 'processed' / 'fund_nav' / 'qixing_fund_nav.parquet')

        nav_adapter = QixingNavAdapter(mode='parity') if _nav_file.exists() else None
        print(f"[INIT] NAV适配器: {'已加载' if nav_adapter else '未加载(文件不存在)'}", flush=True)
        print(f"[INIT] NAV文件: {_nav_file}", flush=True)

        def post_exec_hook(engine):
            """exec() 之后，daily loop 之前，重新注入 wrapper 函数"""
            # ==== 参数注入（优化扫描用，覆盖策略 initialize 的默认值） ====
            # 在所有 patch 之前注入，确保后续 patch（如活跃证券注册）使用最新参数
            if self.param_overrides:
                _g_inject = engine.namespace.get('g')
                if _g_inject is not None:
                    for _k, _v in self.param_overrides.items():
                        setattr(_g_inject, _k, _v)
                        print(f"[SWEEP] 注入 g.{_k} = {_v}", flush=True)

            # ==== 性能优化 1：注册活跃证券集合 ====
            # 策略 initialize() 已执行，g.etf_pool 和 g.defensive_etf 已就绪。
            # 直接设置 _active_securities 而不调用 set_active_securities()，
            # 因为后者会重建 namespace，覆盖已注入的 pd/math/get_name 等。
            _g = engine.namespace.get('g')
            if _g is not None and hasattr(_g, 'etf_pool'):
                _active = list(_g.etf_pool)
                if hasattr(_g, 'defensive_etf') and _g.defensive_etf:
                    if _g.defensive_etf not in _active:
                        _active.append(_g.defensive_etf)
                engine._active_securities = _active
                engine._daily_trade_snapshot_cache.clear()
                engine._daily_current_snapshot_cache.clear()
                print(f"[PERF] 已注册活跃证券集合: {len(_active)} 只 -> {_active}", flush=True)
            else:
                print(f"[PERF] WARNING: g.etf_pool 不可用, _g={_g}", flush=True)

            # ==== 性能优化 2：覆盖 get_name ====
            # 原版 get_name 调用 get_current_data()[sec].name，但引擎 CurrentData
            # 不含 name 属性，每次都触发全市场快照后抛异常被 except 捕获返回"未知"。
            # 改为走 get_security_info 内存缓存，O(1) 返回名称。
            _data_api = engine.data_api
            _name_cache = self._name_cache

            def _fast_get_name(security):
                cached = _name_cache.get(security)
                if cached is not None:
                    return cached
                try:
                    info = _data_api.get_security_info(security)
                    name = info.display_name if info and info.display_name else security
                except Exception:
                    name = security
                _name_cache[security] = name
                return name

            engine.namespace['get_name'] = _fast_get_name

            # ==== Bug 绕过：attribute_history 多字段返回空 + fq='pre' 前复权 ====
            # 引擎 _get_price_impl 在单证券+多字段时调用 _history_cached(field=tuple)
            # 返回空 DataFrame。策略 calculate_momentum_metrics 调用
            # attribute_history(etf, lookback, '1d', ['close', 'high']) 受此影响。
            # 绕过方式：将多字段拆分为单字段请求，然后合并结果。
            # 同时: 聚宽 set_option('use_real_price', True) 启用动态前复权,
            #        策略的 attribute_history 默认返回前复权数据.
            #        local 引擎默认 fq=None (不复权), 导致价格序列在分红派息日附近
            #        产生 0.06+ 级差异, 进而导致短期动量等指标翻转.
            #        例: 511220 04-22, 不复权 close=10.217, 前复权 close=10.152 (差0.065)
            #        这里统一强制 fq='pre'.
            _orig_attr_hist = engine.namespace['attribute_history']

            # ==== Bug修复: load_1d_etf 缓存污染 ====
            # hdata_reader.load_1d_etf 用 _ETF_YEAR_CACHE 按年缓存.
            # 第一次调用若传 columns=['code','date','adj_factor'] (分钟数据 fq='pre' 路径),
            # 只加载6列 (base+extra), 后续全列调用命中缓存拿到部分列 -> close/high/low 全 nan.
            # 修复: monkey-patch load_1d_etf, 始终读全部列 (ETF数据量小, ~39只).
            from core import hdata_reader as _hr_mod
            _orig_load_1d_etf = _hr_mod.load_1d_etf

            def _safe_load_1d_etf(start=None, end=None, codes=None, columns=None):
                # 强制读全部列, 避免部分列污染年度缓存
                return _orig_load_1d_etf(start=start, end=end, codes=codes, columns=None)
            _hr_mod.load_1d_etf = _safe_load_1d_etf
            # 清空可能已被污染的缓存
            _hr_mod._ETF_YEAR_CACHE.clear()
            print(f"[PERF] patched load_1d_etf: 强制全列读取 + 清空已污染缓存", flush=True)

            def _patched_attribute_history(security, count, unit='1d', fields=None,
                                            skip_paused=False, df=True, fq=None,
                                            fq_ref_date=None):
                # 聚宽 use_real_price=True 对应动态前复权, fq=None 时强制改为 'pre'
                if fq is None:
                    fq = 'pre'
                if (fields is None or isinstance(fields, str)
                        or not isinstance(fields, (list, tuple))
                        or len(fields) <= 1):
                    result = _orig_attr_hist(security, count, unit, fields,
                                           skip_paused=skip_paused, df=df,
                                           fq=fq, fq_ref_date=fq_ref_date)
                    return result
                # 多字段：逐字段获取后合并（绕过引擎 _history_cached(field=tuple) 返回空的 bug）
                import pandas as _pd
                pieces = {}
                col = None
                for f in fields:
                    col = _orig_attr_hist(security, count, unit, [f],
                                          skip_paused=skip_paused, df=df,
                                          fq=fq, fq_ref_date=fq_ref_date)
                    if col is None or (hasattr(col, 'empty') and col.empty):
                        return _pd.DataFrame()
                    pieces[f] = col[f].values if f in getattr(col, 'columns', []) else None
                if all(v is None for v in pieces.values()):
                    return _pd.DataFrame()
                result = _pd.DataFrame(pieces)
                result.index = col.index if hasattr(col, 'index') else None
                return result

            engine.namespace['attribute_history'] = _patched_attribute_history
            print(f"[PERF] patched attribute_history: 多字段拆分 + fq='pre' 前复权", flush=True)

            # ==== 对齐修复：511880 佣金 fallback + 501018 LOF 分类 ====
            # 修复3: 511880 命中引擎 temporary_fallbacks.has_zero_fee_fallback 被强制佣金=0,
            #        但策略 set_order_cost(type='fund', open_commission=0.0002, min_commission=5)
            #        明确要收佣金. 聚宽实际收 511880 buy 10000×101.102×0.0002 = 202.2 元佣金.
            #        引擎 temporary_fallbacks.py 注释明确说"temporary", 与策略设置直接冲突.
            # 修复4: 501018 (南方原油LOF) 被引擎 _get_instrument_type 判为 'stock' (50开头不匹配任何分支),
            #        但聚宽按 fund 类型处理, 收 0.0002 佣金. 本地按 stock 收 0.0003, 多收0.0001.
            #        LOF (50开头) 在聚宽属于 fund, 应映射为引擎的 'etf' 类型.
            import sys as _sys_mod2
            _core_mod2 = _sys_mod2.modules.get('engine.core')
            if _core_mod2 is None:
                _core_mod2 = _sys_mod2.modules.get('engine').core

            _orig_has_zero_fee = _core_mod2.has_zero_fee_fallback
            def _patched_has_zero_fee_fallback(security):
                if security == "511880.XSHG":
                    return False
                return _orig_has_zero_fee(security)
            _core_mod2.has_zero_fee_fallback = _patched_has_zero_fee_fallback

            _orig_get_inst_type = engine._get_instrument_type
            def _patched_get_instrument_type(security):
                local_code = security.replace(".XSHE", ".SZ").replace(".XSHG", ".SH").replace(".XBSE", ".BJ")
                clean_code = local_code.split('.')[0]
                if clean_code.startswith('50'):
                    return 'etf'
                return _orig_get_inst_type(security)
            engine._get_instrument_type = _patched_get_instrument_type

            print(f"[PERF] patched has_zero_fee_fallback: 511880→False", flush=True)
            print(f"[PERF] patched _get_instrument_type: 50xxxx→etf (LOF)", flush=True)

            # ==== 对齐修复：price_decimals 对 50 开头 LOF 返回 3 位小数 ====
            # 修复5: 聚宽对 LOF (50开头, 如 501018) 用 3 位小数报价, 与 ETF 一致.
            #        引擎 price_decimals 只识别 51/58/159/56/16/11/12/13/10, 遗漏 50,
            #        返回 2 位小数. 导致 501018 成交价 1.326 被舍入为 1.33 (差 0.004),
            #        进而 last_price 偏高, 短期动量判断翻转, 选股分歧.
            #        jq 交易记录证实 501018 价格均为 3 位小数 (1.334, 1.326 等).
            import engine.order as _order_mod_pd
            import sys as _sys_mod_pd
            _core_mod_pd = _sys_mod_pd.modules.get('engine.core')
            if _core_mod_pd is None:
                _core_mod_pd = _sys_mod_pd.modules.get('engine').core

            _orig_price_decimals = _order_mod_pd.price_decimals
            def _patched_price_decimals(security):
                local_code = security.replace(".XSHE", ".SZ").replace(".XSHG", ".SH").replace(".XBSE", ".BJ")
                clean_code = local_code.split('.')[0]
                if clean_code.startswith('50'):
                    return 3
                return _orig_price_decimals(security)
            _order_mod_pd.price_decimals = _patched_price_decimals
            _core_mod_pd.price_decimals = _patched_price_decimals
            print(f"[PERF] patched price_decimals: 50xxxx→3 (LOF 3位小数)", flush=True)

            # (fq='pre' 前复权修复已合并到上方的 attribute_history patch)
            # patched_get_trade_price 也用 fq='pre' (见下方)

            # ==== 对齐修复：ETF 买单放宽资金充足检查 + 禁用滑点 ====
            # 修复1: 聚宽允许现金为负，本地引擎资金检查砍仓导致少买100股
            # 修复2: 聚宽ETF市价单成交价=基准价(不加滑点), 本地round(slippage)导致价格偏高
            #         例: 4.664+0.0004664=4.6644664, round=4.665, 但聚宽成交4.664
            _orig_execute_trade = engine._execute_trade

            def _patched_execute_trade(order, match_price=None):
                inst_type = engine._get_instrument_type(order.security)
                if inst_type == 'etf':
                    # 临时禁用滑点
                    orig_slip_val = None
                    if hasattr(engine.slippage, 'slippage'):
                        orig_slip_val = engine.slippage.slippage
                        engine.slippage.slippage = 0
                    # 买单: 临时放宽资金检查
                    orig_cash = None
                    if order.amount > 0:
                        orig_cash = engine.context.portfolio.available_cash
                        engine.context.portfolio.available_cash = 1e12
                    try:
                        result = _orig_execute_trade(order, match_price)
                        # 还原现金: after_cash = 1e12 - fill_total_cost
                        if orig_cash is not None:
                            after_cash = engine.context.portfolio.available_cash
                            engine.context.portfolio.available_cash = orig_cash - (1e12 - after_cash)
                        return result
                    finally:
                        if orig_slip_val is not None:
                            engine.slippage.slippage = orig_slip_val
                        if orig_cash is not None and engine.context.portfolio.available_cash > 1e11:
                            # 异常情况, 还原
                            engine.context.portfolio.available_cash = orig_cash
                return _orig_execute_trade(order, match_price)

            engine._execute_trade = _patched_execute_trade

            # ==== 对齐修复：市价单撮合基准价用分钟close而非(O+C+H+L)/4 ====
            # 当前引擎 order.get_trade_price 在日内时段用分钟K线的 (O+C+H+L)/4 作为
            # 市价单基准价，但聚宽用分钟 close。这导致基准价差0.001左右，
            # 进而使 target_amount = int(target_value/price) 在取整边界附近产生
            # 100股级别的数量差异。
            # 修复：monkey-patch engine.order.get_trade_price，日内时段用 close。
            import engine.order as _order_mod

            def _patched_get_trade_price(data_api, current_dt, current_time, security):
                if ':' in str(current_time):
                    parts = str(current_time).split(':')
                    norm_time = f"{int(parts[0]):02d}:{int(parts[1]):02d}"
                else:
                    norm_time = '09:30'

                if norm_time <= "09:35":
                    df = data_api._get_price_raw(
                        security, end_date=current_dt.replace(hour=15, minute=0), count=1,
                        fields=["open", "close", "high_limit", "low_limit", "paused", "volume"],
                        frequency="daily", fq='pre'
                    )
                    field = "open"
                elif norm_time >= "14:55":
                    df = data_api._get_price_raw(
                        security, end_date=current_dt.replace(hour=15, minute=0), count=1,
                        fields=["open", "close", "high_limit", "low_limit", "paused", "volume"],
                        frequency="daily", fq='pre'
                    )
                    field = "close"
                else:
                    # 日内：用14:00的close（策略在14:00决策，聚宽成交价最接近14:00 close）
                    df = data_api._get_price_raw(
                        security, end_date=current_dt.replace(hour=14, minute=1), count=1,
                        fields=["close", "high_limit", "low_limit", "paused", "volume"],
                        frequency="1m", fq='pre'
                    )
                    field = "close"
                    if df.empty:
                        df = data_api._get_price_raw(
                            security, end_date=current_dt.replace(hour=15, minute=0), count=1,
                            fields=["close", "high_limit", "low_limit", "paused", "volume"],
                            frequency="daily", fq='pre'
                        )
                        field = "close"

                if df.empty:
                    return 0, 999999, 0, False, 999999999
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        row = df.xs(security, axis=1, level=1).iloc[0]
                    else:
                        row = df.iloc[0]
                    # 用 Decimal 避免浮点精度问题 (4.6649999 round -> 4.665)
                    # 正常四舍五入, 但用 Decimal 保证精度
                    decimals = _order_mod.price_decimals(security)
                    raw_price = float(row[field])
                    from decimal import Decimal as _Dec, ROUND_HALF_UP as _RHU
                    price = float(_Dec(str(raw_price)).quantize(_Dec('0.1') ** decimals, rounding=_RHU))
                    high_limit = float(_Dec(str(float(row.get("high_limit", 999999)))).quantize(_Dec('0.1') ** decimals, rounding=_RHU))
                    low_limit = float(_Dec(str(float(row.get("low_limit", 0)))).quantize(_Dec('0.1') ** decimals, rounding=_RHU))
                    paused = row.get("paused", False)
                    volume = row.get("volume", 999999999)
                    return price, high_limit, low_limit, paused, volume
                except Exception as e:
                    return 0, 999999, 0, False, 999999999

            _order_mod.get_trade_price = _patched_get_trade_price
            # 同时 patch engine.core 模块中已导入的引用，否则 core.py 调用的还是旧函数
            import sys as _sys_mod
            _core_mod = _sys_mod.modules.get('engine.core')
            if _core_mod is None:
                _core_mod = _sys_mod.modules.get('engine').core
            _core_mod.get_trade_price = _patched_get_trade_price
            print(f"[PERF] patched get_trade_price: order={_order_mod.get_trade_price is _patched_get_trade_price}, core={_core_mod.get_trade_price is _patched_get_trade_price}", flush=True)

            if nav_adapter:
                # 覆盖 get_extras: unit_net_value 走适配器, 其他走引擎
                _engine_get_extras = engine.namespace.get('get_extras')
                def _qixing_get_extras(label, security, start_date=None, end_date=None, **kwargs):
                    if label == 'unit_net_value':
                        return nav_adapter.get_extras(label, security, start_date=start_date, end_date=end_date, **kwargs)
                    return _engine_get_extras(label, security, start_date=start_date, end_date=end_date, **kwargs)
                engine.namespace['get_extras'] = _qixing_get_extras
                engine.namespace['finance'] = nav_adapter.finance
                engine.namespace['query'] = nav_adapter.query

                # ---- 溢价率缺失处理：根据 nav_mode 区分三种语义 ----
                # parity:         NAV 缺失立即终止回测（数据不完整时无法宣称对齐聚宽）
                # realistic:      NAV 缺失返回 None，策略原生逻辑跳过该 ETF（研究/实盘模式）
                # legacy_invalid: 保留旧版 0 溢价兜底（仅用于复现旧污染结果，禁止用于新报告）
                _engine_get_premium = engine.namespace.get('get_premium_rate')
                _nav_missing_log = self._nav_missing_log
                _nav_mode = self.nav_mode

                if _nav_mode == 'legacy_invalid':
                    def _qixing_get_premium_rate(code, date):
                        result = _engine_get_premium(code, date)
                        if result[0] is None:
                            engine.info(f"⚠️ [LEGACY_INVALID] {code} 净值缺失，溢价率兜底为0（结果不可信）")
                            _nav_missing_log.append({'code': code, 'date': str(date), 'mode': 'legacy_invalid'})
                            return (0.0, result[1] or 0, result[2] or 0)
                        return result
                elif _nav_mode == 'parity':
                    def _qixing_get_premium_rate(code, date):
                        result = _engine_get_premium(code, date)
                        if result[0] is None:
                            _nav_missing_log.append({'code': code, 'date': str(date), 'mode': 'parity'})
                            raise RuntimeError(
                                f"[PARITY] NAV 缺失：{code} 在 {date} 无净值数据。"
                                f"parity 模式要求 NAV 完整，请补全数据或切换到 realistic 模式。"
                            )
                        return result
                else:  # realistic
                    def _qixing_get_premium_rate(code, date):
                        result = _engine_get_premium(code, date)
                        if result[0] is None:
                            _nav_missing_log.append({'code': code, 'date': str(date), 'mode': 'realistic'})
                            return (None, result[1], result[2])  # 返回 None，策略原生逻辑跳过
                        return result
                engine.namespace['get_premium_rate'] = _qixing_get_premium_rate
            else:
                print(f"⚠️ 净值数据不存在: {_nav_file}")
                print("   请在聚宽运行 '聚宽净值导出脚本.py' 并导入后再运行")
                class _FakeQuery:
                    def __init__(self, t): self.target = t; self._filters = []
                    def filter(self, *a): self._filters=a; return self
                    def order_by(self, *a): return self
                    def limit(self, n): return self
                class _FakeFNV:
                    code='code'; day='day'; net_value='net_value'
                class _FakeFinance:
                    FUND_NET_VALUE = _FakeFNV()
                    def run_query(self, q): return pd.DataFrame()
                engine.namespace['finance'] = _FakeFinance()
                engine.namespace['query'] = _FakeQuery

            # 注入诊断钩子 — 在 order 调用时记录诊断信息
            original_order = engine.order
            def diagnostic_order(security, amount, **kw):
                if amount != 0:
                    self._record_diagnostic(engine, security, 'order', dict(
                        security=security,
                        requested_amount=amount,
                    ))
                return original_order(security, amount, **kw)
            engine.namespace['order'] = diagnostic_order

            # 关键：将 namespace 变更同步到 sys.modules['jqdata']
            _sync_namespace_to_jqdata(engine)

            # ==== 得分模式 monkey-patch（方向E：选股逻辑改进） ====
            if self.score_mode != 'baseline':
                _patch_score_mode(engine, self.score_mode)
                print(f"[SCORE] 已启用得分模式: {self.score_mode}", flush=True)

            # ==== P0 任务二：NAV 回测前预检 ====
            # 在策略 initialize() 之后、daily loop 之前执行
            # 检查本次 ETF 池 × 回测区间的 NAV 覆盖完整性
            precheck_report = self._run_nav_precheck(engine)
            self._precheck_report = precheck_report
            if precheck_report['should_abort']:
                raise RuntimeError(
                    f"[NAV_PRECHECK] 回测区间内 NAV 覆盖不完整，parity 模式不允许运行。\n"
                    f"缺失 ETF: {precheck_report['missing_codes']}\n"
                    f"请补全 NAV 数据或切换到 realistic 模式（fail-closed）。"
                )

        engine.post_exec_hook = post_exec_hook

        self.engine = engine

        _t0 = time.monotonic()
        engine.run()
        _elapsed = time.monotonic() - _t0

        results = self._collect_results(engine)
        results['elapsed_seconds'] = round(_elapsed, 2)
        results['nav_precheck'] = getattr(self, '_precheck_report', None)
        return results

    def _record_diagnostic(self, engine, security, callback_name, extra=None):
        """记录单次筛选/下单的诊断快照"""
        ctx = engine.context
        today = ctx.current_dt.date() if hasattr(ctx.current_dt, 'date') else ctx.current_dt

        row = {
            'date': str(today),
            'time': str(engine.current_time),
            'callback': callback_name,
            'security': security,
        }

        try:
            cd = engine.get_current_data()
            try:
                cdo = cd[security]
                row['last_price'] = getattr(cdo, 'last_price', 'MISSING')
                row['high_limit'] = getattr(cdo, 'high_limit', 'MISSING')
                row['low_limit'] = getattr(cdo, 'low_limit', 'MISSING')
                row['paused'] = getattr(cdo, 'paused', 'MISSING')
                row['day_open'] = getattr(cdo, 'day_open', 'MISSING')
            except Exception:
                row['last_price'] = 'MISSING'

            port = ctx.portfolio
            row['cash_before'] = round(port.available_cash, 2)
            row['total_value'] = round(port.total_value, 2)

            pos = port.positions.get(security)
            if pos:
                row['position_amount'] = pos.total_amount
                row['position_cost'] = round(pos.avg_cost, 4)
            else:
                row['position_amount'] = 0
                row['position_cost'] = 0

            row['slippage_model'] = type(engine.slippage).__name__
            if hasattr(engine.slippage, 'slippage'):
                row['slippage_param'] = engine.slippage.slippage
            else:
                row['slippage_param'] = 'MISSING'

            try:
                nd = engine.data_api.get_extras('unit_net_value', security,
                                                 start_date=today, end_date=today)
                if nd.empty:
                    row['fund_net_value'] = 'MISSING'
                else:
                    row['fund_net_value'] = round(float(nd.get(security, pd.Series([None])).iloc[0]), 4) if security in nd.columns else 'MISSING'
            except Exception:
                row['fund_net_value'] = 'MISSING'

            row['premium_rate'] = 'MISSING'

        except Exception as e:
            row['diag_error'] = str(e)

        if extra:
            row.update(extra)

        self.diag_records.append(row)

    def _collect_results(self, engine):
        """收集回测结果"""
        trades = []
        for t in engine.trades:
            time_str = t.get('time', '')
            parts = time_str.split() if ' ' in time_str else (time_str, '')
            trades.append({
                'date': parts[0],
                'time': parts[1] if len(parts) > 1 else '',
                'security': t.get('code', ''),
                'amount': t.get('amount', 0),
                'price': t.get('price', 0),
                'commission': round(float(t.get('commission', 0)), 2),
                'tax': round(float(t.get('tax', 0)), 2),
            })

        daily = []
        for s in engine.daily_portfolio_stats:
            pos_str = str(s.get('positions', s.get('holdings', {})))
            daily.append({
                'date': str(s.get('date', '')),
                'cash': round(float(s.get('cash', s.get('available_cash', 0))), 2),
                'value': round(float(s.get('value', s.get('total_value', 0))), 2),
                'positions': pos_str,
            })

        # P0-5: NAV 覆盖率统计
        nav_report = self._build_nav_report()

        return {
            'trades': trades,
            'daily': daily,
            'diag': self.diag_records,
            'final_value': round(engine.context.portfolio.total_value, 2),
            'nav_report': nav_report,
            'nav_mode': self.nav_mode,
            'invalid_due_to_nav_fail_open': self._invalid_due_to_nav_fail_open,
        }

    def _build_nav_report(self):
        """构建 NAV 覆盖率报告：每只 ETF 的缺失次数、缺失日期范围、受影响交易"""
        if not self._nav_missing_log:
            print(f"\n[NAV] ✅ 无净值缺失事件，溢价过滤全程正常工作")
            return {'total_missing': 0, 'by_code': {}, 'affected_pool': []}

        from collections import Counter, defaultdict
        by_code_count = Counter(e['code'] for e in self._nav_missing_log)
        by_code_dates = defaultdict(list)
        for e in self._nav_missing_log:
            by_code_dates[e['code']].append(e['date'])

        print(f"\n{'=' * 70}")
        print(f"[NAV] 净值缺失统计（fail-closed 模式：缺失→跳过该 ETF）")
        print(f"{'=' * 70}")
        print(f"总缺失次数: {len(self._nav_missing_log)}")
        print(f"{'代码':<16} {'缺失次数':>8} {'首次缺失':>12} {'末次缺失':>12}")
        print(f"{'-' * 70}")
        for code in sorted(by_code_count.keys()):
            dates = sorted(by_code_dates[code])
            print(f"{code:<16} {by_code_count[code]:>8} {dates[0]:>12} {dates[-1]:>12}")
        print(f"{'=' * 70}")

        # 判断池中受影响的 ETF（即整个回测区间都缺失 NAV 的）
        affected_pool = sorted(by_code_count.keys())
        if affected_pool:
            print(f"[NAV] ⚠️ 以下 ETF 因净值缺失被溢价过滤跳过（可能影响选股）:")
            print(f"      {affected_pool}")
            print(f"[NAV] 如这些 ETF 在候选池中，回测结果可能不可靠，需补全 NAV 数据")

        return {
            'total_missing': len(self._nav_missing_log),
            'by_code': {code: {'count': by_code_count[code],
                               'first_date': min(by_code_dates[code]),
                               'last_date': max(by_code_dates[code])}
                        for code in by_code_count},
            'affected_pool': affected_pool,
        }

    def _run_nav_precheck(self, engine):
        """P0 任务二：NAV 回测前预检

        在回测开始前，对本次 ETF 池 × 回测区间做完整 NAV 覆盖检查。
        生成预检报告并写入 qixing_optimize/audits/ 目录。
        """
        import pandas as pd
        from pathlib import Path

        # 获取本次回测的 ETF 池（风险池 + 防御 ETF）
        _g = engine.namespace.get('g')
        if _g is None or not hasattr(_g, 'etf_pool'):
            return {'should_abort': False, 'reason': 'g.etf_pool 不可用，跳过预检'}

        risk_pool = list(_g.etf_pool)
        defensive_etf = getattr(_g, 'defensive_etf', None)
        all_pool = list(risk_pool)
        if defensive_etf and defensive_etf not in all_pool:
            all_pool.append(defensive_etf)

        # 加载 NAV 数据
        nav_file = (Path(os.environ.get('HDATA_ROOT', r'D:\Work Space\HData'))
                     / 'data' / 'processed' / 'fund_nav' / 'qixing_fund_nav.parquet')
        if not nav_file.exists():
            return {'should_abort': False, 'reason': f'NAV 文件不存在: {nav_file}'}

        nav_df = pd.read_parquet(nav_file)
        nav_col = 'nav_date' if 'nav_date' in nav_df.columns else 'date'
        nav_df[nav_col] = pd.to_datetime(nav_df[nav_col])

        # 回测区间内的交易日（策略在 14:01 买入时需要前一交易日的 NAV）
        # 简化：用回测区间内的所有日历日做检查
        start_dt = pd.Timestamp(self.start_date)
        end_dt = pd.Timestamp(self.end_date)

        per_code = {}
        missing_codes = []
        for code in all_pool:
            sub = nav_df[nav_df['code'] == code]
            is_defensive = (code == defensive_etf)
            is_risk = code in risk_pool

            if len(sub) == 0:
                # 该 ETF 完全没有 NAV 数据
                per_code[code] = {
                    'in_risk_pool': is_risk,
                    'is_defensive': is_defensive,
                    'required_days': 'all',
                    'valid_days': 0,
                    'coverage_pct': 0.0,
                    'first_valid_date': None,
                    'last_valid_date': None,
                    'missing_days': -1,  # -1 表示完全无数据
                    'first_20_missing': [],
                }
                if is_risk:  # 风险池中缺失才记录
                    missing_codes.append(code)
                continue

            # 计算回测区间内的覆盖
            in_range = sub[(sub[nav_col] >= start_dt) & (sub[nav_col] <= end_dt)]
            # 全部可用 NAV 日期范围
            first_valid = sub[nav_col].min()
            last_valid = sub[nav_col].max()

            # 估算所需天数（回测区间内的交易日）
            # 粗略估计：区间日历日 * 5/7 * 0.97（扣除节假日）
            calendar_days = (end_dt - start_dt).days + 1
            est_trading_days = int(calendar_days * 5 / 7 * 0.97)
            valid_in_range = len(in_range)
            coverage = valid_in_range / est_trading_days * 100 if est_trading_days > 0 else 0

            # 缺失日期（回测区间内需要的但 NAV 没有的）
            missing_count = max(0, est_trading_days - valid_in_range)

            per_code[code] = {
                'in_risk_pool': is_risk,
                'is_defensive': is_defensive,
                'required_days': est_trading_days,
                'valid_days': valid_in_range,
                'coverage_pct': round(coverage, 1),
                'first_valid_date': str(first_valid.date()) if pd.notna(first_valid) else None,
                'last_valid_date': str(last_valid.date()) if pd.notna(last_valid) else None,
                'missing_days': missing_count,
                'first_20_missing': [],  # 详细缺失日期需要交易日历，这里简化
            }
            # 覆盖率 < 90% 的风险池 ETF 标记为缺失
            if is_risk and coverage < 90:
                missing_codes.append(code)

        should_abort = (self.nav_mode == 'parity' and len(missing_codes) > 0)

        report = {
            'should_abort': should_abort,
            'nav_mode': self.nav_mode,
            'backtest_range': f"{self.start_date} ~ {self.end_date}",
            'risk_pool_size': len(risk_pool),
            'defensive_etf': defensive_etf,
            'total_pool_size': len(all_pool),
            'missing_codes': missing_codes,
            'missing_count': len(missing_codes),
            'per_code': per_code,
        }

        # 打印预检摘要
        print(f"\n[NAV_PRECHECK] 模式={self.nav_mode}  区间={self.start_date}~{self.end_date}")
        print(f"[NAV_PRECHECK] 风险池 {len(risk_pool)} 只 + 防御 {defensive_etf} = 总 {len(all_pool)} 只")
        print(f"[NAV_PRECHECK] NAV 覆盖不足的风险池 ETF: {len(missing_codes)} 只 -> {missing_codes}")
        if should_abort:
            print(f"[NAV_PRECHECK] ❌ parity 模式下 NAV 不完整，回测将被终止")
        else:
            print(f"[NAV_PRECHECK] ✅ 允许继续运行")

        # 写入审计文件
        self._write_nav_audit_files(report)

        return report

    def _write_nav_audit_files(self, report):
        """将 NAV 预检报告写入 qixing_optimize/audits/ 目录"""
        import json
        from pathlib import Path

        audit_dir = Path(__file__).parent / 'qixing_optimize' / 'audits'
        audit_dir.mkdir(parents=True, exist_ok=True)

        # JSON 文件
        json_path = audit_dir / 'nav_coverage.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # Markdown 报告
        md_path = audit_dir / 'NAV_COVERAGE_REPORT.md'
        lines = [
            "# NAV 覆盖率预检报告",
            "",
            f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"> 回测区间：{report['backtest_range']}",
            f"> NAV 模式：{report['nav_mode']}",
            f"> 风险池大小：{report['risk_pool_size']}",
            f"> 防御 ETF：{report['defensive_etf']}",
            f"> 总池大小：{report['total_pool_size']}",
            "",
            "## 覆盖率明细",
            "",
            "| ETF 代码 | 风险池 | 防御 | 所需天数 | 有效天数 | 覆盖率% | 首个有效日 | 最后有效日 | 缺失天数 |",
            "|----------|--------|------|---------|---------|---------|-----------|-----------|---------|",
        ]
        for code in sorted(report['per_code'].keys()):
            c = report['per_code'][code]
            risk = "✓" if c['in_risk_pool'] else ""
            defn = "✓" if c['is_defensive'] else ""
            lines.append(
                f"| {code} | {risk} | {defn} | {c['required_days']} | {c['valid_days']} | "
                f"{c['coverage_pct']} | {c['first_valid_date']} | {c['last_valid_date']} | {c['missing_days']} |"
            )
        lines.extend([
            "",
            f"## 预检结论",
            "",
            f"- NAV 覆盖不足的风险池 ETF 数量：{report['missing_count']}",
            f"- 缺失 ETF 列表：{report['missing_codes']}",
            f"- 允许继续运行：{'否' if report['should_abort'] else '是'}",
            f"- 运行模式：{report['nav_mode']}",
        ])
        if report['should_abort']:
            lines.append(f"\n> ⚠️ parity 模式下 NAV 不完整，回测被终止。请补全 NAV 数据或切换到 realistic 模式。")

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"[NAV_PRECHECK] 审计报告已写入: {md_path}")
        print(f"[NAV_PRECHECK] 审计 JSON 已写入: {json_path}")


def _patch_score_mode(engine, score_mode):
    """
    方向E：选股逻辑改进 — 替换 calculate_momentum_metrics 的得分计算部分。
    保留原版的所有过滤逻辑（停牌/盈利保护/成交量/短期动量/跌幅），仅修改最终 score。

    支持模式：
    - multi_period: score = 0.7 * score_25d + 0.3 * score_60d（多周期动量复合）
    - vol_adjusted: score = annualized * r² / volatility（波动率惩罚，Sharpe-like）
    """
    import numpy as np
    import math

    _ns = engine.namespace
    _g = _ns.get('g')
    _orig_func = _ns.get('calculate_momentum_metrics')
    if _orig_func is None:
        print("[SCORE] WARNING: calculate_momentum_metrics 未找到，跳过 patch", flush=True)
        return

    # 多周期模式的长期窗口
    long_lookback = 60

    def _patched_calculate_momentum_metrics(context, etf):
        """复用原版逻辑，仅在得分计算后覆盖 score"""
        try:
            name = _ns['get_name'](etf)
            lookback = max(_g.lookback_days, _g.short_lookback_days) + 20
            # multi_period 需要更长历史
            if score_mode == 'multi_period':
                lookback = max(lookback, long_lookback + 20)
            prices = _ns['attribute_history'](etf, lookback, '1d', ['close', 'high'])
            if len(prices) < _g.lookback_days:
                return None

            current_price = _ns['get_current_data']()[etf].last_price
            price_series = np.append(prices["close"].values, current_price)

            # ===== 1-5: 复用原版过滤逻辑 =====
            if _ns['check_profit_protection'](etf, context):
                return None

            if _g.enable_volume_check:
                vol_ratio = _ns['get_volume_ratio'](context, etf)
                if vol_ratio is not None:
                    annualized = _ns['get_annualized_returns'](price_series, _g.lookback_days)
                    if annualized > _g.volume_return_limit:
                        return None

            if len(price_series) >= _g.short_lookback_days + 1:
                short_return = price_series[-1] / price_series[-(_g.short_lookback_days + 1)] - 1
                short_annualized = (1 + short_return) ** (250 / _g.short_lookback_days) - 1
            else:
                short_annualized = 0

            if _g.use_short_momentum_filter and short_annualized < _g.short_momentum_threshold:
                return None

            # ===== 长期动量计算（原版 baseline 得分） =====
            recent = price_series[-(_g.lookback_days + 1):]
            y = np.log(recent)
            x = np.arange(len(y))
            weights = np.linspace(1, 2, len(y))
            slope, intercept = np.polyfit(x, y, 1, w=weights)
            annualized_returns = math.exp(slope * 250) - 1

            # R²（趋势稳定性）— 与原版一致用 np.mean(y)
            ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
            ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0

            baseline_score = annualized_returns * r_squared

            # ===== 根据 score_mode 计算 最终 score =====
            if score_mode == 'multi_period':
                # 多周期动量复合：25日 + 60日 加权
                if len(price_series) >= long_lookback + 1:
                    recent_long = price_series[-(long_lookback + 1):]
                    yl = np.log(recent_long)
                    xl = np.arange(len(yl))
                    wl = np.linspace(1, 2, len(yl))
                    sl, il = np.polyfit(xl, yl, 1, w=wl)
                    ann_long = math.exp(sl * 250) - 1
                    ss_res_l = np.sum(wl * (yl - (sl * xl + il)) ** 2)
                    ss_tot_l = np.sum(wl * (yl - np.average(yl, weights=wl)) ** 2)
                    r2_long = 1 - ss_res_l / ss_tot_l if ss_tot_l != 0 else 0
                    score_long = ann_long * r2_long
                    final_score = 0.7 * baseline_score + 0.3 * score_long
                else:
                    final_score = baseline_score  # 数据不足时回退
            elif score_mode == 'vol_adjusted':
                # 波动率惩罚：score = annualized * r² / volatility
                if len(price_series) >= _g.lookback_days + 1:
                    daily_returns = np.diff(price_series[-(_g.lookback_days + 1):]) / price_series[-(_g.lookback_days + 1):-1]
                    vol = np.std(daily_returns)
                    final_score = baseline_score / vol if vol > 0 else baseline_score
                else:
                    final_score = baseline_score
            else:
                final_score = baseline_score

            # ===== 近3日跌幅过滤 =====
            if len(price_series) >= 4:
                day1 = price_series[-1] / price_series[-2]
                day2 = price_series[-2] / price_series[-3]
                day3 = price_series[-3] / price_series[-4]
                if min(day1, day2, day3) < _g.loss:
                    return None

            return {
                'etf': etf,
                'etf_name': name,
                'annualized_returns': annualized_returns,
                'r_squared': r_squared,
                'score': final_score,
                'current_price': current_price,
                'short_annualized': short_annualized,
            }
        except Exception as e:
            _ns['log'].debug(f"{etf} 计算动量指标异常: {e}")
            return None

    _ns['calculate_momentum_metrics'] = _patched_calculate_momentum_metrics


# ---- main ----
if __name__ == '__main__':
    start = sys.argv[1] if len(sys.argv) > 1 else '2024-01-01'
    end = sys.argv[2] if len(sys.argv) > 2 else '2024-01-10'

    print(f"=== 七星高照 V1.7 母版 本地对齐（性能优化版）===")
    print(f"区间: {start} ~ {end}\n")
    runner = QixingParityRunner(start, end)

    try:
        results = runner.run()
    except Exception as e:
        print(f"\n!!! 运行失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n=== 交易记录 ({len(results['trades'])} 笔) ===")
    for t in results['trades']:
        print(f"  {t['date']} {t['time']} {t['security']} {'买' if t['amount']>0 else '卖'} {abs(t['amount'])}股 价格{t['price']} 佣金{t['commission']} 税{t['tax']}")

    print(f"\n=== 每日权益 ===")
    for d in results['daily'][:10]:
        print(f"  {d['date']}: 总值 {d['value']}  现金 {d['cash']}  持仓 {d['positions']}")

    print(f"\n最终权益: {results['final_value']}")
    print(f"总耗时: {results['elapsed_seconds']}s")

    # 输出引擎日志（帮助诊断策略执行问题）
    _logs = runner.engine.logs if runner.engine else []
    print(f"\n=== 引擎日志 ({len(_logs)} 条，前60条) ===")
    for l in _logs[:60]:
        print(f"  {l}")

    # 保存结果
    import json
    diag_file = os.path.join(os.path.dirname(__file__), 'results', f'qixing_local_{start}_{end}.json')
    os.makedirs(os.path.dirname(diag_file), exist_ok=True)
    with open(diag_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {diag_file}")
