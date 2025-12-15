import warnings
import os
from tqdm import tqdm
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer
from sklearn.utils.validation import check_X_y, check_array
import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicTransformer

warnings.filterwarnings("ignore")

# ================= 0. 兼容性修复 (Monkey Patch 增强版) =================
# 修复 scikit-learn >= 1.6 与 gplearn 的核心兼容性问题
if not hasattr(SymbolicTransformer, '_validate_data'):
    def _validate_data(self, X, y=None, y_numeric=False, **kwargs):
        if y is not None:
            X, y = check_X_y(X, y, **kwargs)
        else:
            X = check_array(X, **kwargs)

        if not hasattr(self, 'n_features_in_'):
            if X.ndim == 2:
                self.n_features_in_ = X.shape[1]
            else:
                self.n_features_in_ = 1
        return X, y
    SymbolicTransformer._validate_data = _validate_data

#   配置项
Input_file='sz100.csv'
Output_file='sz100_GP_Factor.csv'
Target_period = 5  # 预测 5 日后收益
GP_generation = 20  # 进化代数
GP_population = 3000  # 种群大小
GP_components = 5  # 保留 Top 10 因子



# ================= 1. 传统因子构建 (扩充版) =================
def add_traditional_factors(df):
    print("正在构建传统因子库 (含K线结构、量价相关性)...")
    data = df.copy()
    # 保持code这一列是str类型的;
    data['code'] = data['code'].astype(str)

    # --- A. Momentum ---
    for t in [5, 10, 20, 60]:
        data[f'ROC_{t}'] = data.groupby('code')['close'].pct_change(periods=t)

    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    data['RSI_14'] = data.groupby('code')['close'].transform(lambda x: calculate_rsi(x, 14))

    # BIAS
    for t in [5, 20, 60]:
        ma = data.groupby('code')['close'].transform(lambda x: x.rolling(t).mean())
        data[f'BIAS_{t}'] = (data['close'] - ma) / ma

    # --- B. Volatility ---
    data['RET'] = data.groupby('code')['close'].pct_change()
    for t in [5, 20, 60]:
        data[f'STD_{t}'] = data.groupby('code')['RET'].transform(lambda x: x.rolling(t).std())

    # ATR (绝对值，用于计算 NATR)
    high_low = data['high'] - data['low']
    high_close = (data['high'] - data['close'].shift()).abs()
    low_close = (data['low'] - data['close'].shift()).abs()
    tr_series = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    data['ATR_14'] = tr_series.groupby(data['code']).transform(lambda x: x.rolling(14).mean())
    data['NATR_14'] = data['ATR_14'] / data['close']  # 归一化 ATR

    # --- C. Volume ---
    data['VOL_CHANGE'] = data.groupby('code')['volume'].pct_change()

    # VWAP (绝对价格，后续剔除)
    data['VWAP_D'] = data['amount'] / (data['volume'] * 100 + 1e-9)
    data['VWAP_REL'] = data['close'] / data['VWAP_D']  # 归一化 VWAP

    data['VOL_STD_20'] = data.groupby('code')['volume'].transform(lambda x: (x / x.rolling(20).mean()))

    # --- 【新增】 量价相关性 (CORR_PV) ---
    # 逻辑：计算过去10天收盘价与成交量的相关系数 (-1 到 1)
    # 1 表示量价齐升，-1 表示量价背离
    # reset_index(level=0, drop=True) 是为了对齐 groupby 后的索引
    print("  >> 计算量价相关性...")
    data['CORR_PV_10'] = data.groupby('code').apply(
        lambda x: x['close'].rolling(10).corr(x['volume'])
    ).reset_index(level=0, drop=True)

    # --- D. Trend ---
    ema12 = data.groupby('code')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = data.groupby('code')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())

    # ✅ 修正版 (去量纲，除以收盘价)
    data['MACD'] = (ema12 - ema26) / data['close']
    data['MACD_SIGNAL'] = data.groupby('code')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    data['MACD_HIST'] = data['MACD'] - data['MACD_SIGNAL']

    # --- E. Reversal & Structure ---
    data['HL_PCT'] = (data['high'] - data['low']) / data['close']
    data['CO_PCT'] = (data['close'] - data['open']) / data['open']

    # --- 【新增】 K线形态结构 (Shadows & Body) ---
    # 分母：日内振幅 (加微小值防止除以0)
    range_hl = (data['high'] - data['low']).replace(0, np.nan).fillna(1e-9)

    # 1. 上影线力度: (High - Max(Open, Close)) / Range
    data['SHADOW_UP'] = (data['high'] - data[['open', 'close']].max(axis=1)) / range_hl

    # 2. 下影线力度: (Min(Open, Close) - Low) / Range
    data['SHADOW_DOWN'] = (data[['open', 'close']].min(axis=1) - data['low']) / range_hl

    # 3. 实体力度: Abs(Close - Open) / Range
    data['BODY_ABS'] = (data['close'] - data['open']).abs() / range_hl

    data.dropna(inplace=True)

    # ================= 🚨 核心修正步骤 🚨 =================
    # 在返回数据前，必须剔除掉那些作为中间计算步骤的“绝对值”列
    # 否则 GP 会利用它们进行“作弊”（利用价格绝对值大小选股）
    cols_to_drop = ['ATR_14', 'VWAP_D']

    # 安全删除（检查列是否存在）
    existing_cols_to_drop = [c for c in cols_to_drop if c in data.columns]
    if existing_cols_to_drop:
        data.drop(columns=existing_cols_to_drop, inplace=True)
        print(f"  >> 已剔除有量纲干扰因子: {existing_cols_to_drop}")

    return data


# ================= 🚨 核心修改: 定义并注册自定义算子 =================
def get_custom_functions():
    """
    定义 gplearn 的自定义逻辑算子和非线性算子
    """

    # 1. Signed Square: 保持符号的平方
    # 逻辑: x * |x|。这比 x^2 好，因为它保留了方向（负收益变得更负，正收益变得更正）。
    def _signed_square(x):
        return np.sign(x) * (np.abs(x) ** 2)

    # make_function 将普通 python 函数转换为 gplearn 可用的算子
    # arity=1 表示这个算子接受 1 个参数
    signed_square = make_function(function=_signed_square, name='signed_square', arity=1)

    # 2. If_Else (Ternary Operator)
    # 逻辑: 如果 Condition > 0，则返回 A，否则返回 B
    # 类似于 pandas 的 where 或 numpy 的 where
    def _if_else(condition, true_val, false_val):
        return np.where(condition > 0, true_val, false_val)

    # arity=3 表示这个算子接受 3 个参数 (Condition, A, B)
    if_else = make_function(function=_if_else, name='if_else', arity=3)

    return [signed_square, if_else]

def GP_Dig_Factor(df, feature_cols, target_cols):
    # 建议把这里增加到 50
    print(f"正在挖掘 GP Alpha 因子 (Target={target_cols}, Gens={GP_generation})...")

    # 1. 严格时序划分 (保持你原来的修正)
    unique_dates = df['date'].sort_values().unique()
    split_date = unique_dates[int(len(unique_dates) * 0.7)]
    print(f"【严格时序划分】训练集截止日期: {pd.to_datetime(split_date).strftime('%Y-%m-%d')}")

    train_df = df[df['date'] <= split_date].copy()

    # ========================== 核心修改 A: 输入特征 ==========================
    # ❌ 原代码 (已注释): 只用了原始价格，量纲混乱
    # X_dict = {}
    # base_features = ['open', 'high', 'low', 'close', 'volume', 'amount']
    # ... (省略原代码) ...

    # ✅ 新代码: 使用你计算好的传统因子 (RSI, ROC, BIAS 等)
    # 这些因子已经是比率(Ratio)了，非常适合 GP 组合
    # 如果你想保留原始量价的 lag，可以把它们也加进 feature_cols，但在 main 函数里加
    X = train_df[feature_cols].copy()
    # 简单清洗：处理传统因子可能产生的 inf 或 nan
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    # ========================================================================
    y = train_df[target_cols].fillna(0.5).values

    # ================= 🚨 核心修改: 将算子加入 GP =================
    # 1. 获取自定义算子
    custom_funcs = get_custom_functions()

    # 2. 定义基础算子 (保留你原来用的一些)
    base_functions = ['add', 'sub', 'mul', 'div', 'abs', 'neg', 'inv', 'max', 'min']

    # 3. 合并成完整的 function_set
    function_set = base_functions + custom_funcs

    est = SymbolicTransformer(
        generations=40,  # ✅ 建议调大到 50
        population_size=6000,  # ✅ 保持 2000-3000
        hall_of_fame=100,
        n_components=GP_components,
        function_set=function_set,
        # ========================== 核心修改 B: 惩罚系数 ==========================
        parsimony_coefficient=0.001,  # ✅ 调小100倍！鼓励公式变长、变复杂
        # ========================================================================
        max_samples=0.6,
        # 2. 降低竞争压力 (让非主流因子也能存活，防止同质化)
        tournament_size=3,  # 默认是20，调小一点增加多样性
        # 3. 调整进化概率 (组合拳：少杂交，多突变，多精简)
        p_crossover=0.1,  # 降低杂交 (防止近亲繁殖)
        p_subtree_mutation=0.7,  # 增加新逻辑注入 (引入新思路)
        p_hoist_mutation=0.1,  # 【抗套娃神器】专门对抗 sin(sin(...)) 这种结构
        p_point_mutation=0.1,  # 节点微调 (比如把 max 变成 min)

        verbose=1,
        random_state=42,
        n_jobs=-1  # ✅ 开启多核并行，加速运算
    )

    print(f"开始训练 (Feature Shape: {X.shape})...")
    est.fit(X, y)

    print("GP 挖掘完成，Best Alphas:")
    for i, expr in enumerate(est._best_programs):
        if i < GP_components:
            print(f"Alpha_{i}: {expr}")

    # 应用到全量数据 (同样修正为使用 feature_cols)
    full_X = df[feature_cols].copy()
    full_X = full_X.replace([np.inf, -np.inf], np.nan).fillna(0)

    new_features = est.transform(full_X)

    for i in range(new_features.shape[1]):
        df[f'GP_ALPHA_{i}'] = new_features[:, i]

    return df

# ================= 3. 预处理工程 =================
def preprocess_factors(df, feature_cols):
    print("正在进行截面预处理 (Winsorize -> Z-Score)...")
    processed_df = df.copy()

    def process_day(group):
        cols = [c for c in group.columns if c in feature_cols]
        for col in cols:
            series = group[col]
            # Winsorize，去掉1%的最大值和1%的最小值
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            series = series.clip(lower, upper)
            # Z-Score，标准化
            std = series.std()
            if std != 0:
                series = (series - series.mean()) / std
            else:
                series = 0
            group[col] = series
        return group

    tqdm.pandas(desc="Cross-Section Scaling")
    processed_df = processed_df.groupby('date').progress_apply(process_day)

    return processed_df


# ================= 主程序 =================
def main():
    if not os.path.exists(Input_file):
        print(f"错误: 找不到 {Input_file}")
        return

    print(f"读取数据: {Input_file}...")
    df = pd.read_csv(Input_file, dtype={'code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    # 1. 计算原始 5日收益率
    df['RET_FWD_5'] = df.groupby('code')['close'].shift(-Target_period) / df['close'] - 1

    # 【新增逻辑】 2. 计算 5日收益率的排名 (截面排名)
    # pct=True 表示百分比排名(0~1之间)，method='first' 处理平局情况
    # 必须按 date 分组，因为排名是每天所有股票之间的比较
    print("正在计算截面收益率排名 (Rank)...")
    df['RET_FWD_5_RANK'] = df.groupby('date')['RET_FWD_5'].transform(lambda x: x.rank(pct=True, method='first'))

    # 构建传统因子
    df = add_traditional_factors(df)

    # 确定哪些列需要保留，哪些是因子
    base_cols = ['date', 'code', 'name', 'open', 'high', 'low', 'close', 'volume', 'amount', 'RET_FWD_5',
                 'RET_FWD_5_RANK', 'RET']
    traditional_factors = [c for c in df.columns if c not in base_cols]
    print(f"已构建 {len(traditional_factors)} 个传统因子。")

    # 对df进行预处理，去重
    df_clean = df.dropna().copy()

    print("特征索引对照表:")
    for i, col in enumerate(traditional_factors):
        print(f"X{i}: {col}")

    # 【重要选择】
    # 这里的 target_col 可以选择 'RET_FWD_5' (预测数值) 也可以选择 'RET_FWD_5_RANK' (预测排名)
    # 建议使用 RANK，因为因子的排序能力通常比数值预测能力更重要
    # 这里我改为了传入 'RET_FWD_5_RANK'
    df_with_gp = GP_Dig_Factor(df_clean, traditional_factors, 'RET_FWD_5_RANK')

    # 这里获取gp_factors;就是带有ALPHA
    gp_factors = [c for c in df_with_gp.columns if 'GP_ALPHA' in c]
    all_factors = traditional_factors + gp_factors
    print(f"已挖掘 {len(gp_factors)} 个 GP 因子。")

    # 预处理
    final_df = preprocess_factors(df_with_gp, all_factors)
    final_df.dropna(inplace=True)

    print(f"保存结果至: {Output_file}")
    final_df.to_csv(Output_file, index=False)

    print("特征工程全部完成。")
    print(f"最终维度: {final_df.shape}")


if __name__ == "__main__":
    main()

