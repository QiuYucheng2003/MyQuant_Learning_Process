import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
# 算出 GP_ALPHA_0 (原始), GP_ALPHA_0_EMA_5 (优化) 等所有因子的 ICIR。
# 对于每一个GP_ALPHA_0的家族，进行内部比较，选出参数最好的那个平滑版本，抛弃了不稳定的原始版。
# 最后是去重阶段：选出5个最好的因子中，进行相关性比较，保留 ICIR 更高的那个
# 你手中的 sz100_Final_Selected_Factors.csv 将是非常干净、稳定且低相关的因子集。

# ================= 配置项 =================
Input_file = 'sz100_Optimized_Factor.csv'
Output_file = 'sz100_Final_Selected_Factors.csv'
IC_Report_File = 'Factor_IC_Report.csv'  # 保存详细评测结果

# 评价基准
TARGET_COL = 'RET_FWD_5'  # 使用原始收益率计算 Rank IC
CORRELATION_THRESHOLD = 0.8  # 相关性去重阈值 (超过此值视为重复)

# 绘图配置 (Mac中文显示问题)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ================= 1. 评价函数 =================

def calculate_rank_ic(data, factor_col, target_col):
    """
    计算 Rank IC (斯皮尔曼相关系数)
    返回: Mean IC, IC Std, ICIR
    """
    # 按日期分组，计算因子与收益的秩相关系数
    # 使用 dropna() 确保数据对齐
    daily_ic = data.groupby('date').apply(
        lambda x: x[factor_col].corr(x[target_col], method='spearman')
    )

    mean_ic = daily_ic.mean()
    std_ic = daily_ic.std()
    icir = mean_ic / std_ic if std_ic != 0 else 0

    return mean_ic, icir, daily_ic


def evaluate_all_factors(df, factor_cols):
    """
    对所有因子进行评测
    """
    results = []
    print(f"正在评测 {len(factor_cols)} 个因子...")

    for f in tqdm(factor_cols, desc="Evaluating"):
        mean_ic, icir, _ = calculate_rank_ic(df, f, TARGET_COL)
        results.append({
            'Factor': f,
            'Mean_IC': mean_ic,
            'ICIR': abs(icir),  # 选拔时看绝对值，方向可以反转
            'Direction': 1 if mean_ic > 0 else -1
        })

    return pd.DataFrame(results).sort_values('ICIR', ascending=False)


# ================= 2. 家族选拔与去重 =================

def select_best_variant(eval_df, original_alphas):
    """
    家族内斗：从 Alpha_0 及其所有变体中，选出 ICIR 最高的一个
    """
    best_factors = []

    print("\n========= 阶段一：家族内斗 (Internal Competition) =========")
    for raw_name in original_alphas:
        # 找到该家族所有变体 (比如 GP_ALPHA_0, GP_ALPHA_0_EMA_3...)
        # 逻辑：列名包含 raw_name 且 (是本身 OR 是衍生品)
        family_members = eval_df[eval_df['Factor'].str.contains(raw_name)]

        if family_members.empty:
            continue

        # 选出 ICIR 最大的
        best_one = family_members.loc[family_members['ICIR'].idxmax()]
        best_factors.append(best_one)

        print(
            f"家族 {raw_name} 胜出者: {best_one['Factor']} (ICIR: {best_one['ICIR']:.4f}, IC: {best_one['Mean_IC']:.4f})")

    return pd.DataFrame(best_factors)


def filter_by_correlation(df, candidates, threshold=0.8):
    """
    最终去重：计算胜出者之间的相关性，剔除高度相关的弱势因子
    """
    print(f"\n========= 阶段二：相关性去重 (Threshold={threshold}) =========")

    # 获取这些因子的时序数据 (使用 pivot 转换成 宽表：index=date, columns=factor, values=mean_value_cross_section)
    # 这里我们简化一下：直接计算所有样本的相关性即可，或者计算日均相关性。
    # 为了准确，我们取因子的截面值计算相关性。

    # 提取候选因子的列
    factor_names = candidates['Factor'].tolist()
    data_matrix = df[factor_names]

    # 计算皮尔逊相关系数矩阵
    corr_matrix = data_matrix.corr(method='pearson')

    # 绘制热力图方便查看 (可选)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix Before Filtering')
    plt.tight_layout()
    plt.savefig('Correlation_Matrix.png')
    print("  >> 相关性热力图已保存至 Correlation_Matrix.png")

    # 贪婪剔除法
    # 1. 按 ICIR 从高到低排序
    sorted_candidates = candidates.sort_values('ICIR', ascending=False).reset_index(drop=True)
    kept_factors = []
    dropped_factors = []

    for i in range(len(sorted_candidates)):
        curr_factor_name = sorted_candidates.iloc[i]['Factor']
        curr_icir = sorted_candidates.iloc[i]['ICIR']

        is_correlated = False
        # 和已经保留的因子对比
        for kept_name in kept_factors:
            # 查相关性
            corr_val = abs(corr_matrix.loc[curr_factor_name, kept_name])
            if corr_val > threshold:
                is_correlated = True
                print(f"❌ 剔除 {curr_factor_name}: 与 {kept_name} 高度相关 (Corr={corr_val:.2f})")
                dropped_factors.append(curr_factor_name)
                break

        if not is_correlated:
            kept_factors.append(curr_factor_name)
            print(f"✅ 保留 {curr_factor_name} (ICIR={curr_icir:.4f})")

    return kept_factors


# ================= 主程序 =================

def main():
    if not os.path.exists(Input_file):
        print(f"❌ 找不到文件: {Input_file}")
        return

    print(f"读取数据: {Input_file}...")
    df = pd.read_csv(Input_file, dtype={'code': str})

    # 1. 确定有哪些因子列
    # 排除掉非因子列
    base_cols = ['date', 'code', 'name', 'RET_FWD_5', 'RET_FWD_5_RANK']
    factor_cols = [c for c in df.columns if c not in base_cols and 'GP_ALPHA' in c]

    # 2. 评测所有因子
    eval_report = evaluate_all_factors(df, factor_cols)
    eval_report.to_csv(IC_Report_File, index=False)
    print(f"详细评测报告已保存至 {IC_Report_File}")

    # 3. 确定原始家族列表 (Alpha_0, Alpha_1...)
    # 假设你的原始因子是 GP_ALPHA_0 到 GP_ALPHA_4
    original_families = sorted(list(set([c.split('_EMA')[0].split('_SMA')[0] for c in factor_cols])))
    # 上面这行代码会提取出 ['GP_ALPHA_0', 'GP_ALPHA_1'...]

    # 4. 家族内选拔
    best_variants_df = select_best_variant(eval_report, original_families)

    # 5. 最终相关性去重
    final_factor_names = filter_by_correlation(df, best_variants_df, threshold=CORRELATION_THRESHOLD)

    print("\n========= 最终结果 =========")
    print(f"原始家族数: {len(original_families)}")
    print(f"最终保留数: {len(final_factor_names)}")
    print("最终因子名单:")
    print(final_factor_names)

    # 6. 保存最终文件
    # 只保留 key 列和最终因子列
    final_cols = ['date', 'code', 'name', 'RET_FWD_5'] + final_factor_names
    final_df = df[final_cols]
    final_df.to_csv(Output_file, index=False)
    print(f"\n最终纯净因子文件已保存至: {Output_file}")


if __name__ == "__main__":
    main()