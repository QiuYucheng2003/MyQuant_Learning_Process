import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import os

# 忽略警告
warnings.filterwarnings("ignore")

# ================= 配置参数 =================
INPUT_FILE = 'sz100_Final_Selected_Factors.csv'
OUTPUT_FILE = 'sz100_Final_Composite_Score.csv'

# 核心因子列表 (Alpha_2 最强)
FACTOR_COLS = ['GP_ALPHA_2_EMA_10', 'GP_ALPHA_1_EMA_3', 'GP_ALPHA_3_EMA_10']

# 预测目标
TARGET_LABEL = 'RET_FWD_5_RANK'

# 滚动训练参数 (用于 XGBoost)
TRAIN_WINDOW = 242  # 约1年
TEST_WINDOW = 20  # 约1个月

# Plan B: XGBoost 参数 (进一步简化以防过拟合)
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 20,  # 再次减少树的数量
    'max_depth': 2,  # 极简树深 (类似线性回归加一点点非线性)
    'learning_rate': 0.1,
    'subsample': 0.7,
    'colsample_bytree': 0.5,  # 每次只看一半特征，强制模型利用弱因子
    'n_jobs': -1,
    'random_state': 42
}


# ================= 方法一：简单等权合成 (Plan A) =================
# 逻辑：标准化 -> 平均 -> 变号
# 这是最稳健的基准，往往比复杂的 ML 模型更有效

def calc_simple_composite(df, factors):
    print(f"\n[Plan A] 正在计算简单等权合成 (Factors={len(factors)})...")

    # 临时 DataFrame
    temp_df = df[['date', 'code'] + factors].copy()

    # 定义每日标准化函数
    def process_day(group):
        score_sum = 0
        valid_count = 0

        for col in factors:
            # 强制转数值
            series = pd.to_numeric(group[col], errors='coerce')

            # 截面 Z-Score 标准化 (关键！)
            if series.std() != 0:
                series = (series - series.mean()) / series.std()
            else:
                series = 0

            # 累加 (注意：如果因子 IC 是负的，这里先累加，最后统一变号)
            # 假设所有因子方向一致（都是负向因子）
            score_sum += series

        # 计算平均并取反 (因为原因子 IC 均约为 -0.05，取反后 Score 越大越好)
        return -1 * score_sum / len(factors)

    # Apply (因计算量小，直接 groupby apply 问题不大，或者用 transform 加速)
    # 为了稳健，使用 transform 逐列处理再相加会更快

    # 快速向量化实现：
    final_score = pd.Series(0.0, index=df.index)

    # 按天分组计算每个因子的 Z-Score 并累加
    for col in tqdm(factors, desc="Standardizing"):
        # 转换
        vals = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 计算每日 Mean/Std
        daily_mean = vals.groupby(df['date']).transform('mean')
        daily_std = vals.groupby(df['date']).transform('std').replace(0, 1)  # 防止除0

        # Z-Score
        z_score = (vals - daily_mean) / daily_std

        # 累加
        final_score += z_score

    # 取反并平均
    final_score = -1 * (final_score / len(factors))

    return final_score


# ================= 方法二：XGBoost 原始因子滚动 (Plan B) =================
# 逻辑：不做正交化，直接喂给 XGB，让树模型自己处理共线性

def train_rolling_xgboost_raw(df, feature_cols, target_col):
    print(f"\n[Plan B] 正在训练 XGBoost (Raw Factors, No Ortho)...")
    unique_dates = df['date'].sort_values().unique()
    total_len = len(unique_dates)

    preds = []
    start_idx = 0

    pbar = tqdm(total=total_len, desc="XGBoost Rolling")

    while start_idx + TRAIN_WINDOW < total_len:
        # 切片
        train_start = unique_dates[start_idx]
        test_start = unique_dates[start_idx + TRAIN_WINDOW]
        test_end_idx = min(start_idx + TRAIN_WINDOW + TEST_WINDOW, total_len)
        test_end = unique_dates[test_end_idx - 1]

        if test_start > test_end: break

        train_mask = (df['date'] >= train_start) & (df['date'] < test_start)
        test_mask = (df['date'] >= test_start) & (df['date'] <= test_end)

        if train_mask.sum() < 50:  # 样本过少跳过
            start_idx += TEST_WINDOW
            pbar.update(TEST_WINDOW)
            continue

        # 准备数据 (强制转 float)
        X_train = df.loc[train_mask, feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        y_train = df.loc[train_mask, target_col]
        X_test = df.loc[test_mask, feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

        # 训练
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)

        # 预测
        pred_score = model.predict(X_test)

        # 记录
        temp = df.loc[test_mask, ['date', 'code', 'name', 'RET_FWD_5']].copy()
        temp['SCORE_XGB'] = pred_score
        preds.append(temp)

        start_idx += TEST_WINDOW
        pbar.update(TEST_WINDOW)

    pbar.close()
    if not preds: return pd.DataFrame()
    return pd.concat(preds)


# ================= 主程序 =================

def evaluate_performance(df, score_col, target_col):
    """计算 IC 和 ICIR"""
    df_clean = df.dropna(subset=[score_col, target_col])
    if df_clean.empty: return 0, 0, pd.Series()

    daily_ic = df_clean.groupby('date').apply(
        lambda x: x[score_col].corr(x[target_col], method='spearman')
    )
    return daily_ic.mean(), daily_ic.mean() / daily_ic.std(), daily_ic


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到文件: {INPUT_FILE}")
        return

    print(f"读取因子文件: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, dtype={'code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['date', 'code']).reset_index(drop=True)

    # 补全 Rank
    if TARGET_LABEL not in df.columns:
        print(f"正在计算 {TARGET_LABEL}...")
        df[TARGET_LABEL] = df.groupby('date')['RET_FWD_5'].rank(pct=True, method='first')

    # ------------------------------------------------------
    # 执行 Plan A: 简单合成
    # ------------------------------------------------------
    df['SCORE_SIMPLE'] = calc_simple_composite(df, FACTOR_COLS)

    # ------------------------------------------------------
    # 执行 Plan B: XGBoost (无正交化)
    # ------------------------------------------------------
    xgb_res = train_rolling_xgboost_raw(df, FACTOR_COLS, TARGET_LABEL)

    # 合并结果 (XGBoost 结果比原数据短，因为有滚动窗口期)
    # 我们以 left join 方式合并到主表
    final_df = df.merge(xgb_res[['date', 'code', 'SCORE_XGB']], on=['date', 'code'], how='left')

    # ------------------------------------------------------
    # 最终大比武 (Evaluation)
    # ------------------------------------------------------
    print("\n" + "=" * 40)
    print("🚀 最终合成效果大比武 (Rank IC)")
    print("=" * 40)

    # 评估 Simple
    ic_a, icir_a, daily_ic_a = evaluate_performance(final_df, 'SCORE_SIMPLE', 'RET_FWD_5')
    print(f"[Plan A] 简单等权合成:")
    print(f"   IC Mean: {ic_a:.4f}")
    print(f"   ICIR   : {icir_a:.4f}")

    # 评估 XGBoost
    # 注意：只评估有 XGB 预测值的日期，为了公平对比，Simple 也应该限制在同时间段
    valid_xgb_mask = final_df['SCORE_XGB'].notna()
    df_compare = final_df[valid_xgb_mask].copy()

    if not df_compare.empty:
        ic_b, icir_b, daily_ic_b = evaluate_performance(df_compare, 'SCORE_XGB', 'RET_FWD_5')
        # 重新计算该时间段的 Plan A 以示公平
        ic_a_period, icir_a_period, daily_ic_a_period = evaluate_performance(df_compare, 'SCORE_SIMPLE', 'RET_FWD_5')

        print(f"\n[Plan B] XGBoost (Raw) - 同期对比:")
        print(f"   IC Mean: {ic_b:.4f}")
        print(f"   ICIR   : {icir_b:.4f}")

        print(f"\n[Plan A] 简单合成 (Simple) - 同期对比:")
        print(f"   IC Mean: {ic_a_period:.4f}")
        print(f"   ICIR   : {icir_a_period:.4f}")

        # 自动选择赢家
        winner = 'SCORE_SIMPLE' if icir_a_period > icir_b else 'SCORE_XGB'
        print(f"\n🏆 胜出者: {winner}")

        # 绘图对比
        plt.figure(figsize=(10, 5))
        daily_ic_a_period.cumsum().plot(label=f'Plan A: Simple (ICIR={icir_a_period:.2f})')
        daily_ic_b.cumsum().plot(label=f'Plan B: XGBoost (ICIR={icir_b:.2f})')
        plt.title('Strategy Comparison: Simple vs XGBoost')
        plt.legend()
        plt.grid(True)
        plt.savefig('Strategy_Comparison.png')
        print("对比图已保存至 Strategy_Comparison.png")

        # 将胜出者的分数作为最终 PRED_SCORE
        final_df['PRED_SCORE'] = final_df[winner]

    else:
        print("\n⚠️ XGBoost 尚未生成足够数据，默认使用 Simple Score。")
        final_df['PRED_SCORE'] = final_df['SCORE_SIMPLE']

    # 保存
    out_cols = ['date', 'code', 'name', 'RET_FWD_5', 'SCORE_SIMPLE', 'SCORE_XGB', 'PRED_SCORE']
    final_df[out_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"\n最终结果已保存至: {OUTPUT_FILE}")
    print(f"其中 'PRED_SCORE' 列为自动选出的最佳合成因子。")


if __name__ == "__main__":
    main()