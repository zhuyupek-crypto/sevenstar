"""
七星高照ETF轮动 - 聚宽数据诊断脚本(精简版)
回测区间: 2024-01-01 ~ 2024-12-31, 频率: 每天
"""

def initialize(context):
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_benchmark('518880.XSHG')
    set_slippage(FixedSlippage(0))
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0.0001, close_commission=0.0001, min_commission=0), type='fund')
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')

    g.check_dates = ['2024-01-02', '2024-01-17', '2024-04-23', '2024-07-15']
    g.check_etfs = {
        '2024-01-02': ['513100.XSHG'],
        '2024-01-17': ['511220.XSHG'],
        '2024-04-23': ['511220.XSHG'],
        '2024-07-15': ['501018.XSHG', '161226.XSHE'],
    }
    run_daily(diagnose, time='14:00')


def diagnose(context):
    today = str(context.current_dt.date())
    if today not in g.check_dates:
        return

    log.info('### {} ###'.format(today))
    for sec in g.check_etfs[today]:
        lp = get_current_data()[sec].last_price

        hist = attribute_history(sec, 35, '1d', ['close'])
        closes = hist['close'].values
        ref = closes[-11] if len(closes) >= 11 else 0
        sa = ((lp / ref) ** (25) - 1) if ref > 0 else 0
        sa = (lp / ref) ** (250 / 10) - 1 if ref > 0 else 0

        # 14:00 分钟 close: 用 attribute_history 取 1m 频率
        try:
            hist_m = attribute_history(sec, 1, '1m', ['close'])
            m14 = hist_m['close'][-1] if hist_m is not None and len(hist_m) > 0 else 0
        except Exception:
            m14 = 0

        log.info('{}|last={:.4f}|14m={:.4f}|ref={:.4f}|shortAnn={:.4f}%|{}'.format(
            sec, lp, m14, ref, sa * 100, '过滤' if sa < 0 else '通过'))
