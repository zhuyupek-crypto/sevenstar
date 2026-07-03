# 七星高照 ETF 轮动策略 - 实盘就绪版
# 重构核心：防拥挤、抑滑点、御溢价、散风险
# 底层框架：克隆与吸收 1.7 母版大标的池与动量过滤框架，进行实盘级重构

import numpy as np
import math
import pandas as pd
from jqdata import *

# ==================== 初始化模块 ====================
def initialize(context):
    # ---------- 交易系统设置 ----------
    set_option("avoid_future_data", True)  # 防未来数据
    set_option("use_real_price", True)     # 真实价格交易
    set_slippage(PriceRelatedSlippage(0.0001), type="fund") # 基金交易滑点设置
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0.0002,     # 佣金双边万2
            close_commission=0.0002,
            close_today_commission=0,
            min_commission=5,           # 最低五元佣金限制
        ),
        type="fund",
    )
    set_benchmark("510300.XSHG")       # 基准定为沪深300
    
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    log.set_level('strategy', 'info')
    
    log.info("========== 七星高照 ETF 轮动策略【实盘就绪版】初始化开始 ==========")

    # ---------- 核心参数配置（集中动量模式） ----------
    g.holdings_num = 1                 # 持仓目标数量（集中持有更具爆发力，推荐1）
    g.weight_method = 'equal'          # 资金分配算法：'equal' (等权), 'risk_parity' (波动率反比风险平价)
    g.holding_start_dates = {}         # 记录持仓买入日期，用于实现真实的盘中移动止损（防阴跌）
    g.stop_loss_cooldown = {}          # 记录标的止损时间，实现精准止损冷却期，彻底封死“止损-买回”死循环
    
    # 调仓时间调度
    g.rebalance_frequency = 'weekly'   # 调仓频次：'daily' (每日), 'weekly' (每周重整)
    g.rebalance_weekday = 3            # 若为周频调仓，设定为每周三（避开周一 and 周五的极端情绪波幅）
    g.rebalance_time = '10:30'         # 错峰调仓时间点（上午 10:30，完美避开 14:00 及收盘前散户拥挤踩踏）

    # 过滤限制参数
    g.lookback_days = 25               # 动量评分回看周期
    g.min_daily_turnover = 10000000    # 流动性防线：放宽至 1000 万，释放高动量商品标的（豆粕/原油/白银等）
    g.premium_threshold = 0.05         # QDII溢价率过滤放宽至 5%，允许在合理溢价的大牛市中建仓纳指等标的
    g.loss_threshold = 0.97            # 近3日单日最大跌幅阀值（单日跌超3%直接拉黑）
    g.min_money = 5000                 # 最小交易调仓金额

    # ---------- 盈利保护（Trailing Stop）风控参数 ----------
    g.enable_profit_protection = True                      # 盈利保护开关
    g.profit_protection_threshold = 0.05                   # 盈利保护回撤阈值（高点回撤5%）
    g.profit_protection_check_times = ['11:00', '14:30']   # 每日双时段盘中检查点，防震荡市反复扇巴掌

    # ---------- 标的池配置 ----------
    # 采用 V1.7 大基金池，涵盖大宗商品、海外/港股权益、国内主要风格及指数
    g.etf_pool = [
        # 大宗商品
        "518880.XSHG",  # 黄金ETF
        "159980.XSHE",  # 有色ETF
        "159985.XSHE",  # 豆粕ETF
        "501018.XSHG",  # 南方原油
        "161226.XSHE",  # 白银LOF
        "159981.XSHE",  # 能源化工ETF
        # 国际ETF
        "513100.XSHG",  # 纳指ETF
        "159509.XSHE",  # 纳指科技ETF
        "513290.XSHG",  # 纳指生物ETF
        "513500.XSHG",  # 标普500ETF
        "159529.XSHE",  # 标普消费
        "513400.XSHG",  # 道琼斯ETF
        "513520.XSHG",  # 日经225ETF
        "513030.XSHG",  # 德国30ETF
        "513080.XSHG",  # 法国ETF
        "513310.XSHG",  # 中韩半导体ETF
        "513730.XSHG",  # 东南亚ETF
        # 香港及中概
        "159792.XSHE",  # 港股互联ETF
        "513130.XSHG",  # 恒生科技
        "513050.XSHG",  # 中概互联网ETF
        "159920.XSHE",  # 恒生ETF
        "513690.XSHG",  # 港股红利
        # 国内宽基指数
        "510300.XSHG",  # 沪深300ETF
        "510500.XSHG",  # 中证500ETF
        "510050.XSHG",  # 上证50ETF
        "510210.XSHG",  # 上证ETF
        "159915.XSHE",  # 创业板ETF
        "588080.XSHG",  # 科创50
        "512100.XSHG",  # 中证1000ETF
        "563360.XSHG",  # A500-ETF
        "563300.XSHG",  # 中证2000ETF
        # 风格因子
        "512890.XSHG",  # 红利低波ETF
        "159967.XSHE",  # 创业板成长ETF
        "512040.XSHG",  # 价值ETF
        "159201.XSHE",  # 自由现金流ETF
        # 债券ETF
        "511380.XSHG",  # 可转债ETF
        "511010.XSHG",  # 国债ETF
        "511220.XSHG",  # 城投债ETF
    ]

    # QDII 基金分类池（核心监测对象，防溢价陷阱）
    g.qdii_etfs = {
        "513100.XSHG", "159509.XSHE", "513290.XSHG", "513500.XSHG", 
        "159529.XSHE", "513400.XSHG", "513520.XSHG", "513030.XSHG", 
        "513080.XSHG", "513310.XSHG", "513730.XSHG", "159792.XSHE", 
        "513130.XSHG", "513050.XSHG", "159920.XSHE", "513690.XSHG"
    }

    # 避险防御资产（使用纯货币ETF，锁定0%回撤防线）
    g.defensive_basket = [
        "511880.XSHG",  # 华宝添益货币ETF（纯现金管理工具，年化稳定，0回撤）
    ]

    # ---------- 定时任务调度 ----------
    # 注册每日检查重平衡的错峰时间点
    run_daily(rebalance_check, time=g.rebalance_time)

    # 动态注册盘中盈利保护监控时间点
    if g.enable_profit_protection:
        for check_time in g.profit_protection_check_times:
            run_daily(profit_protection_check, time=check_time)
            log.info(f"已成功加载并注册盘中风控时间点：{check_time}")

    log.info("========== 初始化完成：实盘避险重构机制已启动 ==========")


# ==================== 调仓任务检查与调度 ====================
def rebalance_check(context):
    """
    检查调仓频率：若为周频且在周三，则触发重平衡调仓，错峰上午10:30运行
    """
    today = context.current_dt.date()
    weekday = today.weekday() + 1  # 转换 python 0-6 至 1-7 (周一至周日)
    
    if g.rebalance_frequency == 'daily' or (g.rebalance_frequency == 'weekly' and weekday == g.rebalance_weekday):
        log.info(f"📅 满足重平衡周期要求，正在启动调仓流程（星期{weekday}，系统时间：{g.rebalance_time}）")
        execute_rebalance(context)


# ==================== 重平衡核心交易模块 ====================
def execute_rebalance(context):
    log.info("========== 重平衡调仓开始 ==========")
    
    # 1. 动态过滤并计算最新动量评分
    ranked_etfs = get_ranked_etfs(context)
    
    # 2. 依据动量及正得分阀值提取目标标的
    target_etfs = []
    for item in ranked_etfs:
        if len(target_etfs) >= g.holdings_num:
            break
        if item['score'] > 0:
            target_etfs.append(item['etf'])
            
    log.info(f"🎯 经过多重风控与动量筛选后的候选标的: {target_etfs}")
    
    # 3. 若全市场无正向动量标的，启动多元防守资产包
    is_defensive_mode = False
    if not target_etfs:
        log.info("🛡️ 全市场无安全趋势资产，策略全面撤退，装载多元避险防御包")
        target_etfs = g.defensive_basket
        is_defensive_mode = True
        
    # 4. 计算每只目标资产的目标配置权重
    target_weights = {}
    total_val = context.portfolio.total_value
    
    if is_defensive_mode:
        # 防御资产包均分
        active_defensive = [e for e in target_etfs if check_asset_tradable(e)]
        if not active_defensive:
            log.warning("⚠️ 极端情况：所有避险防御标的均停牌跌停，强制清仓空仓避险")
            for etf in list(context.portfolio.positions.keys()):
                if etf not in g.defensive_basket:
                    smart_order_target_value(etf, 0, context)
            return
        weight = 1.0 / len(active_defensive)
        for etf in active_defensive:
            target_weights[etf] = weight
    else:
        # 动量资产持仓权重分配
        if g.weight_method == 'equal':
            weight = 1.0 / len(target_etfs)
            for etf in target_etfs:
                target_weights[etf] = weight
        elif g.weight_method == 'risk_parity':
            # 风险平价：采用经典日度收益率历史波动率倒数加权法
            volatilities = {}
            total_inv_vol = 0
            for etf in target_etfs:
                h = attribute_history(etf, 21, '1d', ['close'])
                if len(h) >= 20:
                    returns = h['close'].pct_change().dropna()
                    vol = returns.std()
                    # 异常值防御
                    if vol == 0 or np.isnan(vol):
                        vol = 0.02
                else:
                    vol = 0.02  # 默认基准波动率
                volatilities[etf] = vol
                total_inv_vol += 1.0 / vol
                
            for etf in target_etfs:
                target_weights[etf] = (1.0 / volatilities[etf]) / total_inv_vol
                
    # 打印权重配比
    log.info("📊 调仓目标权重配比：")
    for etf, w in target_weights.items():
        log.info(f"  - {etf} {get_name(etf)}: {w*100:.2f}%")
        
    # 5. 执行错峰交易：先清仓非目标 -> 再给目标调低仓位 -> 最后利用释放的现金建仓加仓
    target_set = set(target_weights.keys())
    
    # 5.1 清仓非目标持仓
    for etf in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[etf]
        if pos.total_amount > 0 and etf not in target_set:
            log.info(f"📤 [卖出] 移出持仓池，全额卖出: {etf} {get_name(etf)}")
            if smart_order_target_value(etf, 0, context):
                if etf in g.holding_start_dates:
                    g.holding_start_dates.pop(etf)
            
    # 5.2 减仓：调低多配标的权重，回笼资金
    for etf, w in target_weights.items():
        target_val = total_val * w
        pos = context.portfolio.positions.get(etf, None)
        current_val = pos.total_amount * pos.price if (pos and pos.total_amount > 0) else 0
        if current_val > target_val:
            log.info(f"📉 [降仓] 调整配置比例: {etf} {get_name(etf)} 当前市值 {current_val:.2f} -> 目标 {target_val:.2f}")
            smart_order_target_value(etf, target_val, context)
            
    # 5.3 加仓/买入：完成建仓或加仓操作
    for etf, w in target_weights.items():
        target_val = total_val * w
        pos = context.portfolio.positions.get(etf, None)
        current_val = pos.total_amount * pos.price if (pos and pos.total_amount > 0) else 0
        if current_val < target_val:
            log.info(f"📈 [建仓/加仓] 配置入场: {etf} {get_name(etf)} 当前市值 {current_val:.2f} -> 目标 {target_val:.2f}")
            if smart_order_target_value(etf, target_val, context):
                # 如果是新买入的标的，记录买入起始日期（若已持仓则保留原买入时间，防时间被覆盖）
                if etf not in g.holding_start_dates:
                    g.holding_start_dates[etf] = context.current_dt.date()
            
    log.info("========== 重平衡调仓结束 ==========")


# ==================== 动量核心筛选引擎 ====================
def get_ranked_etfs(context):
    """
    遍历大资产池，执行流动性、溢价、近3日大跌多维物理硬防御过滤，返回正得分动量队列
    """
    ranked_list = []
    # 净值溢价回看前一交易日
    prev_date = get_trade_days(end_date=context.current_dt.date(), count=2)[0]
    
    for etf in g.etf_pool:
        # 1. 交易状态过滤
        if get_current_data()[etf].paused:
            continue
            
        # 2. 流动性防线：排除冷门日均成交额不足5000万的迷你标的，防闪崩及极端滑点
        h_turnover = attribute_history(etf, 5, '1d', ['money'])
        if h_turnover.empty or len(h_turnover) < 5:
            continue
        avg_turnover = h_turnover['money'].mean()
        if avg_turnover < g.min_daily_turnover:
            continue
            
        # 3. QDII跨境溢价排雷（收紧阀值为 1.5%）
        is_qdii = etf in g.qdii_etfs
        premium, price, net = get_premium_rate(etf, prev_date)
        if premium is not None:
            if premium > g.premium_threshold:
                log.info(f"🚨 警报！{etf} {get_name(etf)} 前日溢价率高达 {premium*100:.2f}% > 安全阈值 {g.premium_threshold*100:.1f}%，触发溢价杀风控屏蔽")
                continue
        else:
            # QDII由于海外数据时差，一旦缺失净值溢价率，采取防守原则弃买
            if is_qdii:
                log.info(f"⚠️ {etf} {get_name(etf)} QDII 历史溢价未知，放弃配置以防踩雷")
                continue
                
        # 4. 近3日单日最大跌幅过滤（防高空坠物式暴雷）
        lookback = max(g.lookback_days, 10) + 10
        prices = attribute_history(etf, lookback, '1d', ['close'])
        if len(prices) < g.lookback_days:
            continue
            
        current_price = get_current_data()[etf].last_price
        price_series = np.append(prices["close"].values, current_price)
        
        if len(price_series) >= 4:
            day1 = price_series[-1] / price_series[-2]
            day2 = price_series[-2] / price_series[-3]
            day3 = price_series[-3] / price_series[-4]
            if min(day1, day2, day3) < g.loss_threshold:
                log.info(f"⚠️ {etf} {get_name(etf)} 近3日有单日砸盘暴跌超 {(1-g.loss_threshold)*100:.1f}%，直接拉黑剔除")
                continue
                
        # 5. 止损冷却期风控（防刚被止损的资产立刻被买回，冷却期为 10 个交易日，约2周）
        cooldown_date = g.stop_loss_cooldown.get(etf, None)
        if cooldown_date is not None:
            # 计算自上次止损以来的交易天数
            trade_days_passed = get_trade_days(start_date=cooldown_date, end_date=context.current_dt.date())
            days_count = len(trade_days_passed)
            if days_count < 10:
                log.info(f"🚫 {etf} {get_name(etf)} 处于止损冷却期内（已过 {days_count} 天 < 10天），拒绝买入")
                continue
            else:
                # 冷却期结束，移除记录
                g.stop_loss_cooldown.pop(etf)

        # 6. Clenow 线性对数时间加权回归动量计算
        score = calculate_momentum_score_with_series(price_series, etf, context)
        if score is not None:
            ranked_list.append({
                'etf': etf,
                'score': score
            })
            
    # 降序排列
    ranked_list.sort(key=lambda x: x['score'], reverse=True)
    return ranked_list


def calculate_momentum_score_with_series(price_series, etf, context):
    """
    回归公式：Slope (年化收益) x R² (拟合优度，即上涨稳定性)
    """
    try:
        y = np.log(price_series[-(g.lookback_days + 1):])
        x = np.arange(len(y))
        
        # 线性时间权重，保证近期动量有高响应度
        weights = np.linspace(1, 2, len(y))
        
        # 加权线性拟合
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        annualized_returns = math.exp(slope * 250) - 1
        
        # 算R平方
        ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        
        score = annualized_returns * r_squared
        return score
    except Exception as e:
        log.warning(f"动量计算失败 {etf}: {e}")
        return None


# ==================== 盘中盈利保护实时风控模块 ====================
def profit_protection_check(context):
    """
    盘中实时监控（每天 11:00 和 14:30 执行）
    一旦持仓发生严重高点回撤，不等待收盘，盘中强平并买入防御资产
    """
    if not g.enable_profit_protection:
        return
        
    log.info("========== 盘中盈利保护独立扫描开始 ==========")
    triggered = False
    for sec in list(context.portfolio.positions.keys()):
        # 避险资产池不进行动量止损
        if sec in g.defensive_basket:
            continue
            
        pos = context.portfolio.positions[sec]
        if pos.total_amount > 0:
            if check_profit_protection(sec, context):
                log.info(f"🚨🚨 盘中风控警报！{sec} {get_name(sec)} 触及历史最高点回撤线，立即执行平仓")
                if smart_order_target_value(sec, 0, context):
                    triggered = True
                    if sec in g.holding_start_dates:
                        g.holding_start_dates.pop(sec)
                    # 记录止损日期，启动冷却期防止周三调仓直接买回
                    g.stop_loss_cooldown[sec] = context.current_dt.date()
                    
    # 如果有动量资产在盘中被平仓，应立刻将释放出的资金配置到防御组合中，使资产全天均有配置
    if triggered:
        log.info("📢 启动防御避险机制，配置避险篮子组合")
        deploy_defensive_basket(context)
    log.info("========== 盘中盈利保护扫描完成 ==========")


def check_profit_protection(security, context):
    """
    自买入日期起，核算持仓期历史最高价的动态回撤幅度是否超过止损线（防阴跌）
    """
    threshold = g.profit_protection_threshold
    current_price = get_current_data()[security].last_price
    
    # 获取该资产的买入建仓日期
    start_date = g.holding_start_dates.get(security, None)
    if start_date is None:
        # 若买入时间记录缺失，容灾机制：回看近 30 个交易日的最值
        hist = attribute_history(security, 30, '1d', ['high'])
        max_high = hist['high'].max() if not hist.empty else current_price
    else:
        # 获取自买入日期以来的交易天数，计算此区间的最高价
        trade_days = get_trade_days(start_date=start_date, end_date=context.current_dt.date())
        days_count = len(trade_days)
        if days_count > 0:
            hist = attribute_history(security, days_count, '1d', ['high'])
            max_high = hist['high'].max() if not hist.empty else current_price
        else:
            max_high = current_price

    # 结合盘中最新价更新最高点
    max_high = max(max_high, current_price)
    
    # 判断是否跌破高点一定比例
    if current_price <= max_high * (1 - threshold):
        log.info(f"🔻 {security} {get_name(security)} 触发移动止损：当前价 {current_price:.3f}，自买入高点 {max_high:.3f}，回撤幅度 {(1 - current_price/max_high)*100:.2f}%")
        return True
    return False


def deploy_defensive_basket(context):
    """
    将闲置的可用资金均分入多元防守资产包中
    """
    active_defensive = [e for e in g.defensive_basket if check_asset_tradable(e)]
    if not active_defensive:
        log.warning("避险标的所有停牌，全仓空仓持有现金")
        return
        
    total_val = context.portfolio.total_value
    target_value = total_val / len(active_defensive)
    
    # 再次清扫不该持有的普通标的并清理时间记录，记录止损冷却
    for etf in list(context.portfolio.positions.keys()):
        if etf not in g.defensive_basket:
            pos = context.portfolio.positions[etf]
            if pos.total_amount > 0:
                if smart_order_target_value(etf, 0, context):
                    if etf in g.holding_start_dates:
                        g.holding_start_dates.pop(etf)
                    g.stop_loss_cooldown[etf] = context.current_dt.date()
                
    # 补足防御资产仓位
    for etf in active_defensive:
        smart_order_target_value(etf, target_value, context)


# ==================== 辅助与估算工具 ====================
def get_name(security):
    try:
        return get_current_data()[security].name
    except:
        return "未知"


def check_asset_tradable(security):
    data = get_current_data()
    if data[security].paused:
        return False
    # 避险货币市场基金不作涨跌停判定（防高低价限制数据缺失导致无法买入防守）
    if security == "511880.XSHG":
        return True
    if data[security].last_price >= data[security].high_limit:
        return False
    if data[security].last_price <= data[security].low_limit:
        return False
    return True


def get_premium_rate(code, date):
    """
    查询历史溢价，优先使用Extras净值表，容灾使用Finance基础表格查询
    """
    try:
        # 场内收盘交易价
        price_data = get_price(code, start_date=date, end_date=date, frequency='daily', fields=['close'])
        if price_data.empty:
            return None, None, None
        price = price_data['close'].iloc[0]
        
        # 场外份额净值
        net_data = get_extras('unit_net_value', code, start_date=date, end_date=date, df=True)
        if net_data.empty or pd.isna(net_data[code].iloc[0]):
            q = query(finance.FUND_NET_VALUE).filter(
                finance.FUND_NET_VALUE.code == code,
                finance.FUND_NET_VALUE.day == date
            )
            net_df = finance.run_query(q)
            if not net_df.empty:
                net_value = net_df['net_value'].iloc[0]
            else:
                return None, None, None
        else:
            net_value = net_data[code].iloc[0]
            
        if net_value == 0:
            return None, None, None
            
        premium_rate = (price - net_value) / net_value
        return premium_rate, price, net_value
    except:
        return None, None, None


def smart_order_target_value(security, target_value, context):
    """
    实盘订单防爆防护模块：处理最小调仓额、跌涨停板、T+1锁定、不足手取整
    """
    data = get_current_data()
    name = get_name(security)

    if data[security].paused:
        return False

    price = data[security].last_price
    if price == 0:
        return False

    target_amount = int(target_value / price)
    target_amount = (target_amount // 100) * 100
    if target_amount <= 0 and target_value > 0:
        target_amount = 100

    cur_pos = context.portfolio.positions.get(security, None)
    cur_amount = cur_pos.total_amount if cur_pos else 0
    diff = target_amount - cur_amount
    
    trade_val = abs(diff) * price
    # 最小交易金额检查（清仓除外，防止资产萎缩或止损时无法清仓）
    if target_value > 0 and 0 < trade_val < g.min_money:
        return False

    # 涨跌停保护
    if diff > 0:  # 买入
        if data[security].last_price >= data[security].high_limit:
            log.info(f"🚫 {security} {name} 触发涨停，拒绝高吸买入")
            return False
    elif diff < 0:  # 卖出
        if data[security].last_price <= data[security].low_limit:
            log.info(f"🚫 {security} {name} 触发跌停，拒绝践踏式排单卖出")
            return False
            
    # T+1 可用资金或份额限制
    if diff < 0:
        closeable = cur_pos.closeable_amount if cur_pos else 0
        if closeable == 0:
            log.info(f"🚫 {security} {name} 今日买入锁定，T+1限制下无法在日内调出")
            return False
        diff = -min(abs(diff), closeable)

    if diff != 0:
        order_result = order(security, diff)
        if order_result:
            log.info(f"{'📥 [买入调仓]' if diff > 0 else '📤 [卖出平仓]'} {security} {name} 数量 {abs(diff)} 价格 {price:.3f}")
            return True
        else:
            log.warning(f"❌ 挂单执行异常，券商端反馈退单: {security} {name}")
            return False
    return False
