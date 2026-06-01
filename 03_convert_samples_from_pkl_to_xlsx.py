import os
import gzip
import pickle
import pandas as pd
import openpyxl
import numpy as np
import pathlib
import gzip
from collections import defaultdict
import datetime

def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

def find_samples(case_name:str, i_max:int, base_dir:str, folder_name:str, user_percentage: float=1.0, log_dir: str=None):
    folder_me = []
    iter1_sample_num = 0
    iter2plus_sample_num = 0
    iter2plus_selected_sample_num = 0
    train_files = []

    for i in range(1, 2):
        folder_path = pathlib.Path(base_dir) / f"{case_name}_iter{i}" / f"{folder_name}"
        if folder_path.exists() and folder_path.is_dir():
            pkl_files = list(folder_path.glob("sample_*.pkl"))
            if pkl_files:
                folder_me.append(str(folder_path))
                train_files.extend(str(f) for f in pkl_files)

    iter1_sample_num += len(train_files)

    if not 0.0 < user_percentage <= 1.0:
        raise ValueError("User_percentage must be in (0, 1]")

    instance_groups = defaultdict(list)
    for i in range(2, i_max + 1):
        folder_path = pathlib.Path(base_dir) / f"{case_name}_iter{i}" / folder_name
        if folder_path.exists() and folder_path.is_dir():
            pkl_files = list(folder_path.glob("sample_*.pkl"))
            iter2plus_sample_num += len(pkl_files)
            if pkl_files:
                folder_me.append(str(folder_path))
            if user_percentage == 1.0:
                train_files.extend(str(f) for f in pkl_files)
            else:
                for file_path in pkl_files:
                    try:
                        with gzip.open(file_path, 'rb') as f:
                            sample = pickle.load(f)
                        if 'instance' in sample and 'return' in sample:
                            r = float(sample['return'])
                            instance_key = str(sample['instance'])
                            instance_groups[instance_key].append((r, str(file_path)))
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    logfile_find_samples = os.path.join(log_dir, f'The_second_filter_log.txt')
    if user_percentage == 1.0:
        log(f"iter {i_max}: No second filtering required.", logfile_find_samples)
        log(f"iter {i_max} result: {len(train_files)} samples have been found, including iter1 files:{iter1_sample_num} ", logfile_find_samples)
    else:
        log(f"iter {i_max}: Starting second filter for {len(instance_groups)} instances...", logfile_find_samples)
        for instance_key, group in instance_groups.items():
            returns_before = [r for r, _ in group]
            mean_before = np.mean(returns_before)
            std_before = np.std(returns_before)
            min_before = np.min(returns_before)
            max_before = np.max(returns_before)

            group.sort(key=lambda x: x[0], reverse=True)
            num_selected = max(1, int(len(group) * user_percentage))
            iter2plus_selected_sample_num += num_selected
            selected_group = group[:num_selected]

            returns_after = [r for r, _ in selected_group]
            mean_after = np.mean(returns_after)
            std_after = np.std(returns_after)
            min_after = np.min(returns_after)
            max_after = np.max(returns_after)

            log(
                f"Instance:{instance_key}, "
                f"Before the second filter:sample_num:{len(group)},"
                f"return mean:{mean_before:.4f},"
                f"return std:{std_before:.4f},"
                f"return min:{min_before:.4f},"
                f"return max:{max_before:.4f};"
                f"After the second filter:sample_num:{len(selected_group)},"
                f"return mean:{mean_after:.4f},"
                f"return std:{std_after:.4f},"
                f"return min:{min_after:.4f},"
                f"return max:{max_after:.4f}",
                logfile_find_samples
            )

            train_files.extend(file_path for _, file_path in selected_group)

        log(f"iter {i_max} result: iter1 files:{iter1_sample_num},"
            f"iter2 and beyond: selected files/original files:{iter2plus_selected_sample_num}/{iter2plus_sample_num}", logfile_find_samples)

    return folder_me, train_files

def save_samples_to_excel_with_columns(train_files):
    variable_columns = [
        'BINARY', 'INTEGER', 'IMPLINT', 'CONTINUOUS',
        'coef_normalized', 'has_lb', 'has_ub',
        'sol_is_at_lb', 'sol_is_at_ub', 'sol_frac',
        'basis_status_lower', 'basis_status_basic',
        'basis_status_upper', 'basis_status_zero',
        'reduced_cost', 'age',
        'sol_val', 'inc_val', 'avg_inc_val',
        'p_l_max', 'p_l_min', 'keepT', 'r_l',
        'su_l', 'sd_l',
        'pmax', 'pmin', 'ramp', 'su', 'sd'
    ]

    constraint_columns = [
        'obj_cosine_similarity', 'bias', 'is_tight', 'age', 'dualsol_val_normalized'
    ]

    edge_indices_columns = ['row_idx', 'col_idx']
    edge_features_columns = ['coef_normalized']
    scores_column = ['scores']

    for file_path in train_files:
        if not os.path.isfile(file_path):
            print(f"Warning: {file_path} does not exist, skipping.")
            continue

        try:
            with gzip.open(file_path, 'rb') as f:
                sample = pickle.load(f)

            sample_observation, _, sample_action, sample_action_set, sample_scores = sample['data']
            c_dict, e_dict, v_dict, _ = sample_observation

            constraint_features = c_dict['values']
            edge_indices = e_dict['indices']      # shape: (2, N)
            edge_features = e_dict['values']
            variable_features = v_dict['values']

            var_names = sample.get('var_names', None)

            def to_numpy(x):
                if hasattr(x, 'numpy'):
                    return x.numpy()
                return np.asarray(x)

            constraint_features = to_numpy(constraint_features)
            edge_indices = to_numpy(edge_indices)
            edge_features = to_numpy(edge_features)
            variable_features = to_numpy(variable_features)
            sample_scores = to_numpy(sample_scores)

            if edge_indices.ndim == 2 and edge_indices.shape[0] == 2:
                edge_indices = edge_indices.T
            else:
                print(f"Warning: unexpected shape for edge_indices in {file_path}: {edge_indices.shape}")

            if sample_scores.ndim == 0:
                sample_scores = sample_scores.reshape(1)
            elif sample_scores.ndim > 1:
                sample_scores = sample_scores.flatten()

            if var_names is not None and len(var_names) == len(variable_features):
                df_var = pd.DataFrame(variable_features, columns=variable_columns, index=var_names)
            else:
                if var_names is None:
                    print(
                        f"Info: 'var_names' not found in {os.path.basename(file_path)}, using default integer index.")
                else:
                    print(
                        f"Warning: Length mismatch between var_names ({len(var_names)}) and features ({len(variable_features)}) in {os.path.basename(file_path)}. Using default integer index.")
                df_var = pd.DataFrame(variable_features, columns=variable_columns)

            df_con = pd.DataFrame(constraint_features, columns=constraint_columns)
            df_edge_idx = pd.DataFrame(edge_indices, columns=edge_indices_columns)
            df_edge_feat = pd.DataFrame(edge_features, columns=edge_features_columns)
            df_scores = pd.DataFrame(sample_scores, columns=scores_column)

            base_name = os.path.splitext(file_path)[0]
            excel_path = base_name + '.xlsx'

            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

                df_var.to_excel(writer, sheet_name='variable_features', index=True)

                df_con.to_excel(writer, sheet_name='constraint_features', index=False)
                df_edge_idx.to_excel(writer, sheet_name='edge_indices', index=False)
                df_edge_feat.to_excel(writer, sheet_name='edge_features', index=False)
                df_scores.to_excel(writer, sheet_name='scores', index=False)

            print(f"Saved with column names: {excel_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    train_folder_name, train_files = find_samples("test_case118", 1, "....\data\KIDA_data\samples", "train",
                                                  user_percentage=1.0,log_dir="....\data\KIDA_data\samples")

    save_samples_to_excel_with_columns(train_files)