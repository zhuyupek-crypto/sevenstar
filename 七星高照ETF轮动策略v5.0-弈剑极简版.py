# 克隆/还原自聚宽“弈剑”策略：七星高照5.0最终版（极简防过拟合版）
# 核心思想：摒弃繁复过滤条件，回归动量本质。
# 算法核心：24日加权对数线性回归斜率 × R²（趋势稳定性）
# 风控机制：实时止损（分钟级监控，跌破成本一定比例即卖出）

import numpy as np
import math
import pandas as pd
from jqdata import *

# ==================== 初始化模块 ====================
def initialize(context):
    # ---------- 交易系统设置 ----------
    set_option("avoid_future_data", True)  # 防未来数据
    set_option("use_real_price", True)     # 真实价格
    set_slippage(PriceRelatedSlippage(0.0001), type="fund") # 基金滑点
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0.0002,
            close_commission=0.0002,
            close_today_commission=0,
            min_commission=5,
        ),
        type="fund",
    )
    set_benchmark("510300.XSHG")
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    log.set_level('strategy', 'info')
    
    log.info("========== 七星高照 V5.0 极简版初始化开始 ==========")

    # ---------- 核心标的池（“七星”） ----------
    g.etf_pool = [
        "518880.XSHG",   # 黄金ETF
        "159985.XSHE",   # 豆粕ETF
        "501018.XSHG",   # 南方原油
        "161226.XSHE",   # 白银LOF
        "513100.XSHG",   # 纳指ETF
        "159915.XSHE",   # 创业板ETF
        "511220.XSHG",   # 城投债ETF
    ]
    
    # ---------- 核心参数 ----------
    g.lookback_days = 24               # 动量回看周期（弈剑5.0默认24天）
    g.holdings_num = 1                 # 最大持仓数量（单标的轮动）
    g.defensive_etf = "511880.XSHG"    # 防御标的（华宝添益银华日利等货币ETF）
    g.min_money = 5000                 # 最小交易金额限制
    
    # ---------- 止损参数 ----------
    g.stop_loss_rate = 0.95            # 实时止损比例（跌破买入均价5%）
    
    # ---------- 交易调度 ----------
    # 每天开盘前或盘中进行信号调仓（14:00进行卖出与买入决策）
    run_daily(etf_sell_trade, time='14:00')
    run_daily(etf_buy_trade, time='14:01')
    
    log.info("========== 初始化完成：极简动量轮动机制已启动 ==========")

# ==================== 核心计算模块 ====================
def calculate_momentum_score(etf, context):
    """
    计算 Clenow 动量得分：加权对数回归斜率 x R²
    """
    try:
        # 获取包含当天的历史价格
        prices = attribute_history(etf, g.lookback_days, '1d', ['close'])
        if len(prices) < g.lookback_days:
            return None
            
        current_price = get_current_data()[etf].last_price
        price_series = np.append(prices["close"].values, current_price)
        
        # 对数化
        y = np.log(price_series[-(g.lookback_days + 1):])
        x = np.arange(len(y))
        
        # 施加线性时间权重（越接近当前交易日权重越高，1.0 -> 2.0）
        weights = np.linspace(1.0, 2.0, len(y))
        
        # 加权线性回归
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        annualized_returns = math.exp(slope * 250) - 1
        
        # 判定系数 R²
        ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        
        # 动量得分
        score = annualized_returns * r_squared
        return score
    except Exception as e:
        log.warning(f"计算 {etf} 动量得分出错: {e}")
        return None

def get_ranked_etfs(context):
    """
    遍历标的池，计算得分并进行降序排列
    """
    ranked_list = []
    for etf in g.etf_pool:
        # 停牌过滤
        if get_current_data()[etf].paused:
            continue
            
        score = calculate_momentum_score(etf, context)
        if score is not None and score > 0: # 仅保留动量得分为正的标的
            ranked_list.append({'etf': etf, 'score': score})
            
    # 按得分降序排列
    ranked_list.sort(key=lambda x: x['score'], reverse=True)
    return ranked_list

# ==================== 实时止损模块（弈剑5.0精髓） ====================
def handle_data(context, data):
    """
    分钟级实时监控（handle_data 会在交易时间内每分钟触发一次）
    一旦持仓非防御 ETF 的资产跌破止损线，立即在盘中触发市价或限价单卖出避险
    """
    # 遍历当前持仓
    for etf in list(context.portfolio.positions.keys()):
        # 防御资产不进行动量止损
        if etf == g.defensive_etf:
            continue
            
        pos = context.portfolio.positions[etf]
        if pos.total_amount > 0:
            current_price = get_current_data()[etf].last_price
            # 跌破成本价的 5%
            if current_price < pos.avg_cost * g.stop_loss_rate:
                log.info(f"🚨🚨 实时止损触发！{etf} {get_current_data()[etf].name} 当前价 {current_price:.3f} 跌破成本均价 {pos.avg_cost:.3f} (跌幅超过 { (1 - g.stop_loss_rate)*100:.1f}%)")
                order_target(etf, 0) # 盘中即刻清仓

# ==================== 调仓交易模块 ====================
def etf_sell_trade(context):
    """
    每日 14:00 运行：卖出不再属于目标持仓的 ETF
    """
    log.info("--- 开始每日卖出检查 ---")
    ranked = get_ranked_etfs(context)
    
    # 确定目标资产
    target_etfs = [x['etf'] for x in ranked[:g.holdings_num]]
    
    # 如果动量池全为负数，且防御标的可用，则将防御标的设为目标
    if not target_etfs:
        target_etfs = [g.defensive_etf]
        
    target_set = set(target_etfs)
    
    # 卖出不在目标池的持仓
    for etf in list(context.portfolio.positions.keys()):
        if etf not in target_set:
            pos = context.portfolio.positions[etf]
            if pos.total_amount > 0:
                log.info(f"📤 动量排名移出，卖出持仓: {etf} {get_current_data()[etf].name}")
                order_target(etf, 0)

def etf_buy_trade(context):
    """
    每日 14:01 运行：等权买入目标 ETF，若无动量标的则买入防御标的
    """
    log.info("--- 开始每日买入调仓 ---")
    ranked = get_ranked_etfs(context)
    
    # 获取目标列表
    target_etfs = [x['etf'] for x in ranked[:g.holdings_num]]
    
    if not target_etfs:
        # 无正动量标的，买入防御性资产（避险）
        target_etfs = [g.defensive_etf]
        log.info(f"🛡️ 市场整体走弱，进入防御模式，配置防御标的: {g.defensive_etf}")
        
    # 计算每只 ETF 的目标配置资金
    total_val = context.portfolio.total_value
    target_value = total_val / len(target_etfs)
    
    for etf in target_etfs:
        current_data = get_current_data()
        # 涨停或停牌不买入
        if current_data[etf].paused or current_data[etf].last_price >= current_data[etf].high_limit:
            continue
            
        current_val = 0
        if etf in context.portfolio.positions:
            pos = context.portfolio.positions[etf]
            current_val = pos.total_amount * pos.price
            
        # 5%的成交容差，防频繁小幅调仓
        if abs(current_val - target_value) > target_value * 0.05 or current_val == 0:
            log.info(f"📥 调仓买入/调整: {etf} {current_data[etf].name} 目标市值: {target_value:.2f}")
            order_target_value(etf, target_value)
