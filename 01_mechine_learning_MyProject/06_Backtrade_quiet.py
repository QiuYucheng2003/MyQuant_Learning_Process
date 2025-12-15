import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import os
import warnings

# 忽略 Pandas 的警告
warnings.filterwarnings("ignore")

# ================= 配置参数 =================
PRICE_FILE = 'sz100.csv'
SCORE_FILE = 'sz100_Final_Composite_Score.csv'
START_DATE = '2021-01-01'
END_DATE = '2024-12-31'
INITIAL_CASH = 1000000.0
TOP_K = 30
COMMISSION = 0.0003
STAMP_DUTY = 0.001


# ================= 1. 数据准备 =================
def prepare_data():
    print("正在准备数据 (合并行情与因子分)...")
    if not os.path.exists(PRICE_FILE): raise FileNotFoundError(f"{PRICE_FILE} not found")
    if not os.path.exists(SCORE_FILE): raise FileNotFoundError(f"{SCORE_FILE} not found")

    df_price = pd.read_csv(PRICE_FILE, dtype={'code': str})
    df_price['date'] = pd.to_datetime(df_price['date'])

    df_score = pd.read_csv(SCORE_FILE, dtype={'code': str})
    df_score['date'] = pd.to_datetime(df_score['date'])
    df_score = df_score[['date', 'code', 'PRED_SCORE']]

    df_merge = pd.merge(df_price, df_score, on=['date', 'code'], how='left')
    df_merge['PRED_SCORE'] = df_merge['PRED_SCORE'].fillna(-9999)

    mask = (df_merge['date'] >= pd.to_datetime(START_DATE)) & (df_merge['date'] <= pd.to_datetime(END_DATE))
    df_final = df_merge[mask].copy().sort_values(['date', 'code'])

    print(f"数据准备完成，样本数: {len(df_final)}")
    return df_final


# ================= 2. 数据流 =================
class PandasDataWithScore(bt.feeds.PandasData):
    lines = ('score',)
    params = (
        ('datetime', None),
        ('open', 'open'), ('high', 'high'), ('low', 'low'), ('close', 'close'), ('volume', 'volume'),
        ('score', 'PRED_SCORE'),
        ('openinterest', -1),
    )


# ================= 3. 策略 =================
class TopKStrategy(bt.Strategy):
    params = (
        ('top_k', 10),
        ('print_log', False),
        ('reserve_cash', 0.05)
    )

    def __init__(self):
        self.exec_stats = {
            'buy_success': 0, 'sell_success': 0, 'order_failed': 0,
            'limit_up_skip': 0, 'cash_shortage': 0
        }

    def log(self, txt, dt=None):
        if self.params.print_log:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'[{dt.isoformat()}] {txt}')

    def check_limit_up(self, data):
        try:
            prev_close = data.close[-1]
            open_price = data.open[0]
            if prev_close > 0 and (open_price / prev_close - 1) > 0.098:
                return True
        except:
            return False
        return False

    def next(self):
        candidates = []
        for d in self.datas:
            if len(d) > 0 and d.score[0] > -9000:
                candidates.append((d, d.score[0]))

        candidates.sort(key=lambda x: x[1], reverse=True)
        targets = [x[0] for x in candidates[:self.params.top_k]]
        target_names = set([d._name for d in targets])

        # 1. 先卖
        for d in self.datas:
            if self.getposition(d).size > 0:
                if d._name not in target_names:
                    self.close(data=d)

        # 2. 再买
        current_cash = self.broker.get_cash()
        total_value = self.broker.get_value()
        target_value_per_stock = total_value * (1 - self.params.reserve_cash) / self.params.top_k

        buy_list = []
        for d in targets:
            if self.check_limit_up(d):
                self.exec_stats['limit_up_skip'] += 1
                continue

            pos = self.getposition(d).size
            price = d.open[0]
            if price == 0: continue

            target_size = int((target_value_per_stock / price) / 100) * 100
            if target_size == 0: continue

            diff = target_size - pos
            if diff > 0:
                cost = diff * price * (1 + 0.0003)
                buy_list.append((d, diff, cost))

        for d, size, cost in buy_list:
            if current_cash >= cost:
                self.buy(data=d, size=size)
                current_cash -= cost
            else:
                self.exec_stats['cash_shortage'] += 1

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.exec_stats['buy_success'] += 1
            elif order.issell():
                self.exec_stats['sell_success'] += 1
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.exec_stats['order_failed'] += 1
            print(f"⚠️ 拒单警告: {order.data._name} - {order.getstatusname()}")

    def stop(self):
        print("\n" + "=" * 40)
        print("📊 交易执行统计 (Execution Report)")
        print("=" * 40)
        print(f"✅ 买入成交 : {self.exec_stats['buy_success']} 次")
        print(f"✅ 卖出成交 : {self.exec_stats['sell_success']} 次")
        print(f"⏭️ 涨停跳过 : {self.exec_stats['limit_up_skip']} 次")
        print(f"💰 资金拦截 : {self.exec_stats['cash_shortage']} 次")
        print("-" * 40)
        print(f"❌ 拒单总数 : {self.exec_stats['order_failed']} 次")
        if self.exec_stats['order_failed'] == 0:
            print("\n🎉 恭喜！拒单问题已完美解决！")
        else:
            print("\n⚠️ 依然存在拒单，请检查。")
        print("=" * 40)


# ================= 4. 佣金 =================
class StockCommissionScheme(bt.CommInfoBase):
    params = (('stocklike', True), ('commtype', bt.CommInfoBase.COMM_PERC), ('perc', 0.0003), ('stamp_duty', 0.001),)

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:
            return size * price * self.p.perc
        elif size < 0:
            return -(size * price * (self.p.perc + self.p.stamp_duty))
        return 0.0


# ================= 5. 主程序 =================
def main():
    try:
        df = prepare_data()
    except Exception as e:
        print(e);
        return

    cerebro = bt.Cerebro()
    grouped = df.groupby('code')
    for code, group in grouped:
        group = group.set_index('date')
        data = PandasDataWithScore(dataname=group, plot=False)
        cerebro.adddata(data, name=code)

    cerebro.addstrategy(TopKStrategy, top_k=TOP_K)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.addcommissioninfo(StockCommissionScheme(perc=COMMISSION, stamp_duty=STAMP_DUTY))
    # 滑点模拟设置
    # cerebro.broker.set_slippage_perc(perc=0.001)
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02, timeframe=bt.TimeFrame.Days,
                        annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    print(f"开始回测 (Start: {START_DATE}, End: {END_DATE}, TopK: {TOP_K})...")
    results = cerebro.run()
    strat = results[0]

    # --- 核心修改：计算并打印关键指标 ---
    final_value = cerebro.broker.getvalue()
    total_return = (final_value / INITIAL_CASH) - 1

    # 提取分析器结果
    sharpe_info = strat.analyzers.sharpe.get_analysis()
    drawdown_info = strat.analyzers.drawdown.get_analysis()

    # 安全获取夏普比率 (防止数据不足时报错)
    sharpe_ratio = sharpe_info.get('sharperatio', None)

    # 安全获取最大回撤
    max_drawdown = drawdown_info.get('max', {}).get('drawdown', 0.0)

    # 计算年化收益 (CAGR)
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    years = (end_dt - start_dt).days / 365.25
    cagr = (final_value / INITIAL_CASH) ** (1 / years) - 1 if years > 0 else 0

    print("\n" + "=" * 30)
    print("📈 最终回测报告 (Performance Report)")
    print("=" * 30)
    print(f"初始资金 : {INITIAL_CASH:,.2f}")
    print(f"最终资金 : {final_value:,.2f}")
    print(f"总收益率 : {total_return * 100:.2f}%")
    print("-" * 30)
    print(f"年化收益 : {cagr * 100:.2f}%")
    print(f"夏普比率 : {sharpe_ratio:.4f}" if sharpe_ratio is not None else "夏普比率 : N/A")
    print(f"最大回撤 : {max_drawdown:.2f}%")
    print("=" * 30)

    # 绘图
    import matplotlib.pyplot as plt
    returns = strat.analyzers.timereturn.get_analysis()
    cum_value = (1 + pd.Series(returns).sort_index()).cumprod()
    plt.figure(figsize=(10, 6))
    cum_value.plot(title=f'Backtest Result (Total: {total_return * 100:.1f}%, DD: {max_drawdown:.1f}%)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.grid(True)
    plt.savefig('Backtest_Curve_BT_Quiet.png')
    print("曲线图已保存: Backtest_Curve_BT_Quiet.png")


if __name__ == "__main__":
    main()