import os
import gzip
import pickle
import numpy as np
from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    raise ImportError("Please install pandas: pip install pandas")
try:
    import openpyxl
except ImportError:
    raise ImportError("Please install openpyxl: pip install openpyxl")


def process_files_in_folder(folder_path):
    all_stats = {
        'numfrac': [],
        'equal_bestscore': [],
        'larger_95percent': [],
        'larger_90percent': [],
        'larger_85percent': [],
        'larger_80percent': [],
        'larger_70percent': [],
        'larger_60percent': [],
    }

    all_node_depth = {
        'd_between_0_10': [],
        'd_between_10_20': [],
        'd_between_20_30': [],
        'd_between_30_40': [],
        'd_larger_40': [],
    }

    binary_counts = []
    continuous_counts = []
    total_variable_counts = []
    all_continuous_solval = []
    per_sample_min_list = []
    per_sample_max_list = []
    per_sample_mean_list = []  # The average value of consecutive "solval" for each sample

    constraint_counts = []
    edge_counts = []

    pkl_files = [f for f in os.listdir(folder_path) if f.startswith("sample_") and f.endswith(".pkl")]
    if not pkl_files:
        print(f"The sample_*.pkl was not found in {folder_path}.")
        return

    def extract_number(filename):
        try:
            return int(filename.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            return float('inf')

    pkl_files = sorted(pkl_files, key=extract_number)
    print(f"A total of {len(pkl_files)} sample_*.pkl files were found and processing has begun...\n")

    for filename in tqdm(pkl_files, desc="Processing samples", unit="file"):
        filepath = os.path.join(folder_path, filename)
        try:
            with gzip.open(filepath, 'rb') as f:
                sample = pickle.load(f)

            sample_observation, _, _, _, sample_scores = sample['data']
            c_dict, e_dict, v_dict, norm_data = sample_observation

            values_c = c_dict['values']  # (N, D)
            values_e = e_dict['values']  # (N, D)
            total_cons = values_c.shape[0]
            total_edges = values_e.shape[0]
            constraint_counts.append(total_cons)
            edge_counts.append(total_edges)

            values = v_dict['values']  # (N, D)
            total_vars = values.shape[0]
            total_variable_counts.append(total_vars)

            is_binary = (values[:, 0] == 1)
            is_continuous = (values[:, 3] == 1)

            binary_counts.append(np.sum(is_binary))
            continuous_counts.append(np.sum(is_continuous))

            solval = values[:, 16]
            if np.any(is_continuous):
                cont_solval = solval[is_continuous]
                all_continuous_solval.extend(cont_solval)
                sample_min = np.min(cont_solval)
                sample_max = np.max(cont_solval)
                sample_mean = np.mean(cont_solval)

                per_sample_min_list.append(sample_min)
                per_sample_max_list.append(sample_max)
                per_sample_mean_list.append(sample_mean)

            # --- Best Score Statistics ---
            sample_scores = np.array(sample_scores)
            if sample_scores.size == 0:
                continue

            best_score = np.max(sample_scores)
            numfrac = len(sample_scores)
            equal_bestscore = np.isclose(sample_scores, best_score, atol=1e-8).sum()
            larger_95percent = ((sample_scores >= 0.95 * best_score) & (sample_scores <= best_score)).sum()
            larger_90percent = ((sample_scores >= 0.90 * best_score) & (sample_scores <= best_score)).sum()
            larger_85percent = ((sample_scores >= 0.85 * best_score) & (sample_scores <= best_score)).sum()
            larger_80percent = ((sample_scores >= 0.80 * best_score) & (sample_scores <= best_score)).sum()
            larger_70percent = ((sample_scores >= 0.70 * best_score) & (sample_scores <= best_score)).sum()
            larger_60percent = ((sample_scores >= 0.60 * best_score) & (sample_scores <= best_score)).sum()

            all_stats['numfrac'].append(numfrac)
            all_stats['equal_bestscore'].append(equal_bestscore)
            all_stats['larger_95percent'].append(larger_95percent)
            all_stats['larger_90percent'].append(larger_90percent)
            all_stats['larger_85percent'].append(larger_85percent)
            all_stats['larger_80percent'].append(larger_80percent)
            all_stats['larger_70percent'].append(larger_70percent)
            all_stats['larger_60percent'].append(larger_60percent)

            # --- Node Depth ---
            node_depth = sample['node_depth']
            if 0 <= node_depth <= 10:
                all_node_depth['d_between_0_10'].append(node_depth)
            elif 10 < node_depth <= 20:
                all_node_depth['d_between_10_20'].append(node_depth)
            elif 20 < node_depth <= 30:
                all_node_depth['d_between_20_30'].append(node_depth)
            elif 30 < node_depth <= 40:
                all_node_depth['d_between_30_40'].append(node_depth)
            elif node_depth > 40:
                all_node_depth['d_larger_40'].append(node_depth)

        except Exception as e:
            tqdm.write(f" An error occurred while processing {filename}: {e}")
            continue

    if not all_stats['numfrac']:
        print("No samples were successfully processed.")
        return

    total_samples = len(all_stats['numfrac'])

    # === Basic Statistics ===
    avg_total_vars = np.mean(total_variable_counts) if total_variable_counts else 0.0
    avg_binary = np.mean(binary_counts) if binary_counts else 0.0
    avg_continuous = np.mean(continuous_counts) if continuous_counts else 0.0
    global_solval_mean = np.mean(all_continuous_solval) if all_continuous_solval else np.nan
    avg_min_solval = np.mean(per_sample_min_list) if per_sample_min_list else np.nan
    avg_max_solval = np.mean(per_sample_max_list) if per_sample_max_list else np.nan

    if per_sample_mean_list and not np.isnan(global_solval_mean) and global_solval_mean != 0:
        ratios_mean = [(m - global_solval_mean) / global_solval_mean for m in per_sample_mean_list]
        idx_max_ratio_mean = int(np.argmax(ratios_mean))
        idx_min_ratio_mean = int(np.argmin(ratios_mean))
        max_mean_val = per_sample_mean_list[idx_max_ratio_mean]
        min_mean_val = per_sample_mean_list[idx_min_ratio_mean]
        max_mean_ratio = ratios_mean[idx_max_ratio_mean]
        min_mean_ratio = ratios_mean[idx_min_ratio_mean]
    else:
        max_mean_val = min_mean_val = 0.0
        max_mean_ratio = min_mean_ratio = np.nan

    if per_sample_max_list and not np.isnan(avg_max_solval) and avg_max_solval != 0:
        ratios_max = [(m - avg_max_solval) / avg_max_solval for m in per_sample_max_list]
        idx_max_ratio_max = int(np.argmax(ratios_max))
        idx_min_ratio_max = int(np.argmin(ratios_max))
        max_max_val = per_sample_max_list[idx_max_ratio_max]
        min_max_val = per_sample_max_list[idx_min_ratio_max]
        max_max_ratio = ratios_max[idx_max_ratio_max]
        min_max_ratio = ratios_max[idx_min_ratio_max]
    else:
        max_max_val = min_max_val = 0.0
        max_max_ratio = min_max_ratio = np.nan

    avg_constraints = np.mean(constraint_counts) if constraint_counts else 0.0
    avg_edges = np.mean(edge_counts) if edge_counts else 0.0

    df_summary_text = pd.DataFrame({
        "统计摘要": [
            f"数据集路径: {folder_path}",
            f"共处理样本数: {total_samples}",
            f"平均每个样本的候选解数量 (numfrac): {np.mean(all_stats['numfrac']):.2f}",
            "",
            "=== 变量类型说明 ===",
            "v_dict['values'] 第1~4列（索引0~3）表示变量类型，每行仅一个为1：",
            "  - 索引0（第1列）为1 → 二元变量",
            "  - 索引3（第4列）为1 → 连续变量",
            "solval 取自索引16（第17列）",
            "",
            "=== solval 统计范围 ===",
            "仅对连续变量（索引3==1）的 solval 进行统计"
        ]
    })

    # === Var & Solval 表 ===
    var_solval_data = {
        "统计项": [
            "平均每个样本的变量个数",
            "平均每样本二元变量个数",
            "平均每样本连续变量个数",
            "所有连续变量 solval 的平均值",
            "每样本连续变量 solval 最小值的平均值",
            "每样本连续变量 solval 最大值的平均值",
            "样本连续 solval 均值偏离最大的数值",
            "样本连续 solval 均值偏离最小的数值",
            "样本连续 solval 最大值偏离最大的数值",
            "样本连续 solval 最大值偏离最小的数值",
        ],
        "数值": [
            avg_total_vars,
            avg_binary,
            avg_continuous,
            global_solval_mean,
            avg_min_solval,
            avg_max_solval,
            max_mean_val,
            min_mean_val,
            max_max_val,
            min_max_val,
        ],
        "比例": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            max_mean_ratio,
            min_mean_ratio,
            max_max_ratio,
            min_max_ratio,
        ]
    }
    df_var_solval = pd.DataFrame(var_solval_data)

    def format_value(x):
        if pd.isna(x):
            return "N/A"
        return f"{x:.2f}"

    def format_ratio(x):
        if pd.isna(x):
            return "N/A"
        return f"{x * 100:.3f} %"

    df_var_solval["数值"] = df_var_solval["数值"].apply(format_value)
    df_var_solval["比例"] = df_var_solval["比例"].apply(format_ratio)

    # === BestScore Stats ===
    def compute_avg_and_ratio(key):
        avg_count = np.mean(all_stats[key])
        avg_ratio = np.mean([c / n for c, n in zip(all_stats[key], all_stats['numfrac'])]) * 100
        return avg_count, avg_ratio

    bestscore_data = {
        "指标": [
            "等于 best_score 的候选",
            "≥95% 且 ≤100% best_score 的候选",
            "≥90% 且 ≤100% best_score 的候选",
            "≥85% 且 ≤100% best_score 的候选",
            "≥80% 且 ≤100% best_score 的候选",
            "≥70% 且 ≤100% best_score 的候选",
            "≥60% 且 ≤100% best_score 的候选",
        ],
        "平均个数": [
            compute_avg_and_ratio('equal_bestscore')[0],
            compute_avg_and_ratio('larger_95percent')[0],
            compute_avg_and_ratio('larger_90percent')[0],
            compute_avg_and_ratio('larger_85percent')[0],
            compute_avg_and_ratio('larger_80percent')[0],
            compute_avg_and_ratio('larger_70percent')[0],
            compute_avg_and_ratio('larger_60percent')[0],
        ],
        "平均比例 (%)": [
            compute_avg_and_ratio('equal_bestscore')[1],
            compute_avg_and_ratio('larger_95percent')[1],
            compute_avg_and_ratio('larger_90percent')[1],
            compute_avg_and_ratio('larger_85percent')[1],
            compute_avg_and_ratio('larger_80percent')[1],
            compute_avg_and_ratio('larger_70percent')[1],
            compute_avg_and_ratio('larger_60percent')[1],
        ]
    }
    df_bestscore = pd.DataFrame(bestscore_data)
    df_bestscore["平均个数"] = df_bestscore["平均个数"].map(lambda x: f"{x:.2f}")
    df_bestscore["平均比例 (%)"] = df_bestscore["平均比例 (%)"].map(lambda x: f"{x:.3f} %")

    # === Node Depth ===
    total_files = len(pkl_files)
    node_depth_data = {
        "深度区间": [
            "0 ≤ d ≤ 10",
            "10 < d ≤ 20",
            "20 < d ≤ 30",
            "30 < d ≤ 40",
            "d > 40"
        ],
        "样本数量": [
            len(all_node_depth['d_between_0_10']),
            len(all_node_depth['d_between_10_20']),
            len(all_node_depth['d_between_20_30']),
            len(all_node_depth['d_between_30_40']),
            len(all_node_depth['d_larger_40']),
        ],
        "占比 (%)": [
            len(all_node_depth['d_between_0_10']) / total_files * 100,
            len(all_node_depth['d_between_10_20']) / total_files * 100,
            len(all_node_depth['d_between_20_30']) / total_files * 100,
            len(all_node_depth['d_between_30_40']) / total_files * 100,
            len(all_node_depth['d_larger_40']) / total_files * 100,
        ]
    }
    df_nodedepth = pd.DataFrame(node_depth_data)
    df_nodedepth["样本数量"] = df_nodedepth["样本数量"].map(lambda x: f"{x:.2f}")
    df_nodedepth["占比 (%)"] = df_nodedepth["占比 (%)"].map(lambda x: f"{x:.3f} %")

    df_ce_stats = pd.DataFrame({
        "统计项": [
            "平均每个样本的约束个数",
            "平均每个样本的边（非零系数）个数"
        ],
        "数值": [
            avg_constraints,
            avg_edges
        ]
    })
    df_ce_stats["数值"] = df_ce_stats["数值"].apply(lambda x: f"{x:.2f}")

    # === Save to Excel ===
    dataset_name = os.path.basename(folder_path.rstrip(os.sep))
    output_excel = os.path.join(os.path.dirname(folder_path), f"{dataset_name}_analysis_summary.xlsx")

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_summary_text.to_excel(writer, sheet_name="Overall Summary", index=False)
        df_var_solval.to_excel(writer, sheet_name="Var Type & Solval", index=False)
        df_bestscore.to_excel(writer, sheet_name="BestScore Stats", index=False)
        df_nodedepth.to_excel(writer, sheet_name="Node Depth Dist", index=False)
        df_ce_stats.to_excel(writer, sheet_name="c_e_sheet", index=False)  # <-- 新增工作表

    print(f"\n\t所有统计结果已保存至:\n{output_excel}")


if __name__ == "__main__":
    ## case118
    folder_path = r'...\data\samples\case118_iter1\train_MC_UE'
    process_files_in_folder(folder_path)
    print(f"\n")
    folder_path = r'...\data\KIDA_data\samples\case118_iter1\valid'
    process_files_in_folder(folder_path)

    ## Anonymous
    # folder_path = r'...\data\KIDA_data\samples\Anonymous_iter1\train_MC_UE'
    # process_files_in_folder(folder_path)
    # print(f"\n")
    # folder_path = r'...\data\KIDA_data\samples\Anonymous_iter1\valid'
    # process_files_in_folder(folder_path)

    ## Polish 2383
    # folder_path = r'...\data\KIDA_data\samples\case2383_iter1\train_MC_UE'
    # process_files_in_folder(folder_path)
    # print(f"\n")
    # folder_path = r'...\data\KIDA_data\samples\case2383_iter1\valid'
    # process_files_in_folder(folder_path)

    ## French 1888
    # folder_path = r'...\data\KIDA_data\samples\case1888_iter1\train_MC_UE'
    # process_files_in_folder(folder_path)
    # print(f"\n")
    # folder_path = r'...\data\KIDA_data\samples\case1888_iter1\valid'
    # process_files_in_folder(folder_path)