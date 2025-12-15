import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import os
import warnings

# 忽略 Pandas 的警告
warnings.filterwarnings("ignore")

# ================= 配置参数 =================
PRICE_FILE = 'sz100.csv'  # 原始行情文件 (包含 Open, High, Low, Close, Volume)
SCORE_FILE = 'sz100_Final_Composite_Score.csv'  # 你的打分文件
START_DATE = '2021-01-01'  # 回测开始时间
END_DATE = '2024-12-31'  # 回测结束时间
INITIAL_CASH = 1000000.0  # 初始资金 100万
TOP_K = 30  # 每天持有分数最高的 10 只股票
COMMISSION = 0.0003  # 手续费 (万三)
STAMP_DUTY = 0.001  # 印花税 (千一，仅卖出收)


# ================= 1. 数据准备 =================
def prepare_data():
    print("正在准备数据 (合并行情与因子分)...")

    # 读取行情 (OHLCV)
    if not os.path.exists(PRICE_FILE):
        raise FileNotFoundError(f"找不到行情文件 {PRICE_FILE}")
    df_price = pd.read_csv(PRICE_FILE, dtype={'code': str})
    df_price['date'] = pd.to_datetime(df_price['date'])

    # 读取分数 (PRED_SCORE)
    if not os.path.exists(SCORE_FILE):
        raise FileNotFoundError(f"找不到打分文件 {SCORE_FILE}")
    df_score = pd.read_csv(SCORE_FILE, dtype={'code': str})
    df_score['date'] = pd.to_datetime(df_score['date'])

    # 只需要分数列——PRED_SCORE
    if 'PRED_SCORE' not in df_score.columns:
        raise ValueError("打分文件中缺少 'PRED_SCORE' 列")
    df_score = df_score[['date', 'code', 'PRED_SCORE']]

    # 合并: Left Join (以行情为主，匹配分数)
    df_merge = pd.merge(df_price, df_score, on=['date', 'code'], how='left')

    # 填充空分数: 如果某天某股票没有分数，填充为一个极小值，确保不会被买入
    df_merge['PRED_SCORE'] = df_merge['PRED_SCORE'].fillna(-9999)

    # 筛选回测时间段，确保股票数据都是在START_DATE到END_DATE中的
    mask = (df_merge['date'] >= pd.to_datetime(START_DATE)) & (df_merge['date'] <= pd.to_datetime(END_DATE))
    df_final = df_merge[mask].copy()

    # 必须按日期排序，Backtrader 要求
    df_final = df_final.sort_values(['date', 'code'])

    print(f"数据准备完成，样本数: {len(df_final)}")
    return df_final


# ================= 2. 定义数据流 (PandasData) =================
# 我们需要告诉 Backtrader 哪一列是 Score
class PandasDataWithScore(bt.feeds.PandasData):
    # 增加一条自定义线 'score'
    lines = ('score',)

    # 定义参数，对应 DataFrame 的列名
    # -1 表示自动匹配列名，或者指定列的索引
    params = (
        # date是索引列;
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('score', 'PRED_SCORE'),  # 告诉 BT，score 线读取 PRED_SCORE 列
        ('openinterest', -1),
    )


# ================= 3. 定义策略 (TopK Strategy) =================
class TopKStrategy(bt.Strategy):
    params = (
        ('top_k', 10),  # 持仓数量
        ('print_log', False),  # 是否打印日志
        ('reserve_cash', 0.05)  # 预留 5% 现金做缓冲，防止拒单
    )

    def __init__(self):
        self.orders = {}

    def log(self, txt, dt=None):
        if self.params.print_log:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'[{dt.isoformat()}] {txt}')

    def check_limit_up(self, data):
        # 简易涨停判断：如果开盘价相对于昨收涨幅 > 9.8%，认为是涨停无法买入
        # 注意：Backtrader 的 data.close[-1] 是昨收
        # 这里的逻辑是保守估计，实盘中可能需要更复杂的判断
        try:
            prev_close = data.close[-1]
            open_price = data.open[0]
            if prev_close > 0 and (open_price / prev_close - 1) > 0.098:
                return True
        except:
            return False
        return False

    def next(self):
        # next() 会在每一个时间点（天）被调用一次

        # 1. 选股逻辑
        candidates = []
        for d in self.datas:
            # 确保有数据
            if len(d) > 0:
                score = d.score[0]
                # 过滤掉无效分数 (-9999)
                if score > -9000:
                    candidates.append((d, score))

        # 按分数排序
        candidates.sort(key=lambda x: x[1], reverse=True)

        # 选出 Top K 目标 (只取 data 对象)
        targets = [x[0] for x in candidates[:self.params.top_k]]
        target_names = set([d._name for d in targets])

        # ================= 优化核心：先卖后买 + 整手控制 =================

        # A. 第一步：卖出 (清仓不在目标池的)
        # 必须先执行卖出，才能释放资金给后续的买入
        for d in self.datas:
            pos = self.getposition(d).size
            if pos > 0:  # 当前有持仓
                if d._name not in target_names:
                    # 不在今天的 Top K 里，清仓
                    self.log(f'SELL (Clear): {d._name}, Pos: {pos}')
                    self.close(data=d)

                    # B. 第二步：准备买入
        # 注意：Backtrader 的 order 执行是在次日。
        # 我们这里假设"卖出"释放的资金在今天不可用（T+1逻辑或保守估计），只用当前可用现金。
        # 这样能极大减少 Margin 报错。

        current_cash = self.broker.get_cash()
        total_value = self.broker.get_value()

        # 目标是每只股票占总资产的 (1 - 预留现金) / K
        target_value_per_stock = total_value * (1 - self.params.reserve_cash) / self.params.top_k

        buy_list = []

        for d in targets:
            # 1. 涨停检查
            if self.check_limit_up(d):
                self.log(f'SKIP (Limit Up): {d._name}')
                continue

            pos = self.getposition(d).size
            price = d.open[0]  # 假设按今日开盘价买入

            if price == 0: continue

            # 2. 计算目标股数
            target_size = target_value_per_stock / price
            # 向下取整到 100 股 (A股规则)
            target_size = int(target_size / 100) * 100

            if target_size == 0: continue

            # 3. 计算需要买多少 (目标 - 当前)
            diff = target_size - pos

            # 如果需要买入 (diff > 0)
            if diff > 0:
                # 估算需要现金 (含手续费缓冲)
                cost = diff * price * (1 + 0.0003)
                buy_list.append((d, diff, cost))

            # 如果需要卖出 (diff < 0)
            # 说明虽然还在 TopK，但仓位过重，或者股价涨太多了，需要再平衡
            # 这里简化处理：如果在 TopK 里且有持仓，暂不微调，减少交易磨损
            # elif diff < 0 and pos > 0:
            #     self.sell(data=d, size=abs(diff))

        # C. 第三步：执行买入 (根据现金余额严格控制)
        # 我们按分数高低优先买入
        for d, size, cost in buy_list:
            if current_cash >= cost:
                self.log(f'BUY (Add): {d._name}, Size: {size}, Est.Cost: {cost:.2f}')
                self.buy(data=d, size=size)
                current_cash -= cost  # 扣除预估现金，防止后续没钱
            else:
                self.log(f'CASH SHORTAGE: Cannot buy {d._name}, Need: {cost:.2f}, Have: {current_cash:.2f}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'>> BUY SUCCESS: {order.data._name}, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}')
            elif order.issell():
                self.log(
                    f'>> SELL SUCCESS: {order.data._name}, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'>> Order Failed: {order.data._name} - {order.getstatusname()}')
# ================= 4. 自定义手续费模型 (A股模式) =================
class StockCommissionScheme(bt.CommInfoBase):
    params = (
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('perc', 0.0003),  # 买入万三
        ('stamp_duty', 0.001),  # 印花税千一
    )

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:  # 买入
            return size * price * self.p.perc
        elif size < 0:  # 卖出
            # 卖出 = 佣金 + 印花税
            return -(size * price * (self.p.perc + self.p.stamp_duty))
        return 0.0


# ================= 5. 主程序 =================
def main():
    # 1. 准备大表数据
    try:
        df = prepare_data()
    except Exception as e:
        print(f"数据准备失败: {e}")
        return

    # 2. 初始化 Cerebro 引擎
    cerebro = bt.Cerebro()

    print("正在加载 Data Feeds (这可能需要几秒钟)...")
    # 3. 将数据按股票代码拆分，逐个添加进 Cerebro
    # 这是 Backtrader 处理多只股票的标准方式
    grouped = df.groupby('code')

    for code, group in grouped:
        # 设置索引为 date
        group = group.set_index('date')

        # 创建数据流
        data = PandasDataWithScore(
            dataname=group,
            plot=False  # 关闭单只股票绘图，防止图表太乱
        )

        # 添加数据，并命名为股票代码
        cerebro.adddata(data, name=code)

    print(f"已添加 {len(cerebro.datas)} 只股票的数据流。")

    # 添加滑点
    cerebro.broker.set_slippage_perc(perc=0.001)

    # 4. 添加策略
    cerebro.addstrategy(TopKStrategy, top_k=TOP_K)

    # 5. 设置资金与手续费
    cerebro.broker.setcash(INITIAL_CASH)

    # 使用自定义A股佣金模式
    comminfo = StockCommissionScheme(perc=COMMISSION, stamp_duty=STAMP_DUTY)
    cerebro.broker.addcommissioninfo(comminfo)

    # 6. 添加分析指标 (Analyzers)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02, timeframe=bt.TimeFrame.Days,
                        annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')  # 方便后续通过pyfolio分析

    # 7. 运行回测
    print(f"\n开始回测 (Start: {START_DATE}, End: {END_DATE}, Initial Cash: {INITIAL_CASH:.0f})...")
    results = cerebro.run()
    strat = results[0]

    # 8. 输出结果
    final_value = cerebro.broker.getvalue()
    pnl = final_value - INITIAL_CASH
    ret = (final_value / INITIAL_CASH) - 1

    print("\n" + "=" * 30)
    print("回测结果摘要")
    print("=" * 30)
    print(f"初始资金: {INITIAL_CASH:,.2f}")
    print(f"最终资金: {final_value:,.2f}")
    print(f"净收益  : {pnl:,.2f}")
    print(f"收益率  : {ret * 100:.2f}%")

    # 分析器结果
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()

    print(f"夏普比率: {sharpe['sharperatio'] if 'sharperatio' in sharpe else 'N/A'}")
    print(f"最大回撤: {drawdown['max']['drawdown']:.2f}%")
    print("=" * 30)

    # 9. 简单绘图 (资金曲线)
    # 由于股票太多，直接 plot() 会非常卡且乱，我们只提取资金曲线自己画
    import matplotlib.pyplot as plt

    # 提取每日收益率
    returns = strat.analyzers.timereturn.get_analysis()
    ret_df = pd.Series(returns).sort_index()

    # 计算累计净值
    cum_value = (1 + ret_df).cumprod()

    plt.figure(figsize=(10, 6))
    cum_value.plot(title=f'Backtest Result: Top {TOP_K} Strategy (Total Return: {ret * 100:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value (Start=1.0)')
    plt.grid(True)
    plt.savefig('Backtest_Curve_BT.png')
    print("资金曲线已保存至 Backtest_Curve_BT.png")


if __name__ == "__main__":
    main()