import pandas as pd
import tushare as ts
import numpy as np
import matplotlib.pyplot as plt


# ---------------接口导入------------------
pro = ts.pro_api('12345')

# ---------------数据获取------------------
df = pro.daily(ts_code='000001.SZ', start_date='20220101', end_date='20240620')
df = df.sort_values(by='trade_date').reset_index(drop=True)


'''
策略：
1、当ma5日均线上穿ma20日均线，第二天买入
2、当ma5日均线低于ma20日均线，第二天卖出
注：买卖价格均为当日开盘价
'''
# --------------计算移动平均----------------
# 计算 ma5 和 ma20
df['ma5'] = df['close'].rolling(window=5).mean()
df['ma20'] = df['close'].rolling(window=20).mean()

# --------------制定交易策略-----------------
# 5日均线高于20日的记为1，否则记为-1
df['signal'] = np.where(df['ma5'] > df['ma20'], 1, -1)
# 开仓当天取决于前一天的signal信号，所以需要将signal信号后移一天
df['signal'] = df['signal'].shift(1)

# ------------初始化持仓和资金----------------
initial_cash = 100000.0  # 初始资金 100000
df['position'] = 0.0
df['cash'] = initial_cash
df['portfolio_value'] = initial_cash


# --------------计算每日收益-----------------
# 后面用来计算未使用策略时的累计收益
df['daily_return'] = df['close'].pct_change()

# ----------------模拟交易------------------
for i in range(1, len(df)):
    if df['signal'].iloc[i] == 1:  # 昨天信号是买入
        if df['position'].iloc[i-1] == 0:  # 之前没有持仓
            df.at[i, 'position'] = df['cash'].iloc[i-1] / df['open'].iloc[i]    # 计算持仓股票数
            df.at[i, 'cash'] = 0
        else:  # 如果之前持仓，则继续持仓
            df.at[i, 'position'] = df['position'].iloc[i-1]
            df.at[i, 'cash'] = df['cash'].iloc[i-1]
    elif df['signal'].iloc[i] == -1:  # 昨天信号是卖出
        if df['position'].iloc[i-1] > 0:  # 有持仓
            df.at[i, 'cash'] = df['position'].iloc[i-1] * df['open'].iloc[i]    # 股数*当前开盘价
            df.at[i, 'position'] = 0
        else:  # 继续没有持仓
            df.at[i, 'position'] = 0
            df.at[i, 'cash'] = df['cash'].iloc[i-1]
    # 计算策略总收益（现金cash加上持仓股票数*当前开盘价）
    df.at[i, 'portfolio_value'] = df['cash'].iloc[i] + df['position'].iloc[i] * df['close'].iloc[i]

# ------------计算策略的累计收益率-----------
# 计算策略收益的累乘收益率
df['strategy_return'] = df['portfolio_value'].pct_change()
df['strategy_cum_return'] = (1 + df['strategy_return'].fillna(0)).cumprod()
# 计算股票的累计收益率
df['stock_cum_return'] = (1 + df['daily_return'].fillna(0)).cumprod()


# -----------------输出结果------------------
from backtesting import calculate
strategy = df['strategy_cum_return']
print('='*30)
print('策略：')
calculate.Tongji(strategy,rf=4)
no_strategy = df['stock_cum_return']
print('='*30)
print('无策略：')
calculate.Tongji(no_strategy,rf=4)

# -----------------绘制图像------------------
plt.figure(figsize=(12, 8))
plt.plot(df['trade_date'], df['stock_cum_return'], label='Stock Cumulative Return')
plt.plot(df['trade_date'], df['strategy_cum_return'], label='Strategy Cumulative Return')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.title('Stock vs. Strategy Cumulative Return')
plt.show()

