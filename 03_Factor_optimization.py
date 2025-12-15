import pandas as pd
import numpy as np
import os
from tqdm import tqdm
# 这是一个因子优化的文件，针对每一个 GP_ALPHA_n，生成了 6 个变体：
# _EMA_3, _EMA_5, _EMA_10 (指数平滑，权重递减)
# _SMA_3, _SMA_5, _SMA_10 (简单平均)
# 生成的文件sz100_Optimized_Factor.csv中，多了30个因子；


# ================= 配置项 =================
Input_file = 'sz100_GP_Factor.csv'  # 上一步 GP 挖掘产出的文件
Output_file = 'sz100_Optimized_Factor.csv'  # 输出包含优化后因子的文件

# 需要优化的因子列名 (对应你挖掘出的 Top 5)
# Alpha_0: sqrt(add(X13, min(X3, X5)))
# Alpha_1: add(X13, min(X3, X5))
# Alpha_2: sqrt(add(X13, min(X3, min(add(X7, X7), min(X14, X20)))))
# Alpha_3: sqrt(add(X13, min(X3, X20)))
# Alpha_4: sqrt(sqrt(add(X13, min(X3, X5))))
TARGET_ALPHAS = ['GP_ALPHA_0', 'GP_ALPHA_1', 'GP_ALPHA_2', 'GP_ALPHA_3', 'GP_ALPHA_4']

# 平滑窗口参数 (Days)
# 3日: 快速平滑，保留部分灵敏度
# 5日: 标准平滑，周频节奏
# 10日: 深度平滑，过滤掉大部分噪音
WINDOWS = [3, 5, 10]


# ================= 优化函数定义 =================

def process_optimization(df):
    """
    对指定的 GP 因子列进行时序平滑处理
    """
    data = df.copy()

    # 必须先按标的和时间排序
    data = data.sort_values(['code', 'date'])

    print(f"正在对 {len(TARGET_ALPHAS)} 个原始因子进行平滑优化...")

    # 使用 groupby 避免循环，加速计算
    grouped = data.groupby('code')

    for alpha_col in tqdm(TARGET_ALPHAS, desc="Optimizing Alphas"):
        if alpha_col not in data.columns:
            print(f"⚠️ 警告: 找不到列 {alpha_col}，跳过。")
            continue

        # 1. EMA 平滑 (Exponential Moving Average) - 推荐
        # span=W 意味着主要权重集中在最近 W 天
        for w in WINDOWS:
            new_col_name = f"{alpha_col}_EMA_{w}"
            # transform 保持索引一致
            data[new_col_name] = grouped[alpha_col].transform(lambda x: x.ewm(span=w, adjust=False).mean())

        # 2. SMA 平滑 (Simple Moving Average) - 对比用
        # 简单均值，最平稳但滞后严重
        for w in WINDOWS:
            new_col_name = f"{alpha_col}_SMA_{w}"
            data[new_col_name] = grouped[alpha_col].transform(lambda x: x.rolling(window=w, min_periods=1).mean())

    return data


def main():
    # 1. 读取数据
    if not os.path.exists(Input_file):
        print(f"❌ 错误: 找不到输入文件 {Input_file}")
        return

    print(f"读取数据: {Input_file}...")
    df = pd.read_csv(Input_file, dtype={'code': str})
    df['date'] = pd.to_datetime(df['date'])

    # 2. 执行优化
    df_optimized = process_optimization(df)

    # 3. 整理输出
    # 我们保留原始列 + 新生成的优化列 + 基础信息列
    # 基础列
    base_cols = ['date', 'code', 'name', 'RET_FWD_5', 'RET_FWD_5_RANK']

    # 所有因子列 (Raw + Optimized)
    factor_cols = [c for c in df_optimized.columns if 'GP_ALPHA' in c]

    # 合并
    final_cols = base_cols + factor_cols
    # 确保列存在
    final_cols = [c for c in final_cols if c in df_optimized.columns]

    df_final = df_optimized[final_cols]

    # 4. 保存
    print(f"原始因子数量: {len(TARGET_ALPHAS)}")
    print(f"优化衍生后因子总数: {len(factor_cols)}")
    print(f"保存结果至: {Output_file}")
    df_final.to_csv(Output_file, index=False)
    print("✅ 因子优化完成！请进行下一步评价与去重。")


if __name__ == "__main__":
    main()