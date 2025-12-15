import akshare as ak
import pandas as pd
import time
from tqdm import tqdm  # 修正导入方式，通常这样导入更方便
import os
# 需要警醒的点:
# append 操作不会原地修改 DataFrame，它会返回一个新的对象。你必须写成 df = df.append(new)。
# 在循环中使用 append 效率极低（O(N^2)复杂度），数据量大时会极其卡顿。使用列表 list 存储每一个 stock 的 DataFrame，最后使用 pd.concat 一次性合并。

# ================= 1. 网络设置 (解决 ProxyError) =================
# 强制禁用代理，确保国内数据源连接顺畅
os.environ['http_proxy'] = ""
os.environ['https_proxy'] = ""
os.environ['all_proxy'] = ""
os.environ['no_proxy'] = "*"

# 配置
Index_code = '000016'  # 【修正1】上证50的标准代码是 000016
Start_time = '20200101'  # akshare通常建议由纯数字组成的字符串YYYYMMDD，虽然带横杠有时也行，但统一格式更好
End_time = '20241231'
Out_File = 'sz50_data.csv'
Adjust = "qfq"


def get_sz50_stocks():
    print(f"正在获取 {Index_code} 成分股列表...")
    try:
        # 获取上证指数成分股
        df_stocks = ak.index_stock_cons(symbol=Index_code)
        # 只获取 代码 和 名称这两列
        df_stocks = df_stocks[['品种代码', '品种名称']]
        # 【修正2】直接赋值来修改列名
        df_stocks.columns = ['asset_id', 'name']
        print(f"sz50的股票数量为{len(df_stocks)}")
        return df_stocks
    except Exception as e:
        print(f"获取成分股失败, 失败原因: {e}")
        return pd.DataFrame()


def get_and_clean_SingleStock(asset_id, name, start_date, end_date):
    attempt_times = 5
    for i in range(attempt_times):
        try:
            # 下载股票数据
            single_stock = ak.stock_zh_a_hist(symbol=asset_id, start_date=start_date, end_date=end_date, adjust=Adjust)

            if single_stock is None or single_stock.empty:
                return None

            # 重命名
            rename_map = {'日期': 'date', '开盘': 'open', '收盘': 'close',
                          '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount'}
            single_stock.rename(columns=rename_map, inplace=True)

            # 数据清洗
            single_stock['date'] = pd.to_datetime(single_stock['date'])

            num_columns = ['open', 'close', 'high', 'low', 'volume', 'amount']
            for column in num_columns:
                if column in single_stock.columns:
                    single_stock[column] = pd.to_numeric(single_stock[column], errors='coerce')

            # 去掉停牌和空数据
            single_stock = single_stock[(single_stock['volume'] > 0) & (single_stock['open'].notna())]

            if single_stock.empty:
                return None

            # 添加 asset_id 列
            single_stock['asset_id'] = asset_id

            # 只保留需要的列
            # 注意：akshare返回的数据列可能不包含所有列，这里做一个交集处理防止报错
            cols_to_keep = ['asset_id', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']
            existing_cols = [c for c in cols_to_keep if c in single_stock.columns]
            single_stock = single_stock[existing_cols]

            return single_stock

        except Exception as e:
            if i < attempt_times - 1:
                time.sleep(0.1)
                continue
            else:
                # 只有最后一次失败才打印错误
                print(f"Error fetching {asset_id} ({name}): {e}")
                return None


def main():
    # 1. 获取上证50成分股
    sz50_stocks = get_sz50_stocks()
    if sz50_stocks.empty:
        print("未获取到成分股信息，程序退出。")
        return

    # 【修正3】使用列表来暂存数据，而不是 DataFrame.append
    stock_data_list = []

    # 使用 tqdm 显示进度条
    for index, row in tqdm(sz50_stocks.iterrows(), total=sz50_stocks.shape[0], desc="下载进度"):
        asset_id = row['asset_id']
        name = row['name']  # 【修正4】这里原来写成了 asset_id

        # [去 ST] 简单的名称过滤
        if "ST" in name:
            continue

        stock = get_and_clean_SingleStock(asset_id, name, Start_time, End_time)

        if stock is not None and not stock.empty:
            stock_data_list.append(stock)  # 添加到列表中

        time.sleep(1)

        # 3. 合并构建 Panel Data
    if stock_data_list:  # 判断列表是否为空
        print("正在合并数据...")
        panel_data = pd.concat(stock_data_list, ignore_index=True)

        # 排序
        panel_data.sort_values(by=['asset_id', 'date'], inplace=True)
        panel_data.reset_index(drop=True, inplace=True)

        # 保存
        panel_data.to_csv(Out_File, index=False, encoding='utf-8-sig')
        print(f"上证50数据获取完成！共获取 {len(panel_data)} 条数据，已保存至 {Out_File}")
    else:
        print("未获取到任何有效的股票数据。")


if __name__ == '__main__':
    main()