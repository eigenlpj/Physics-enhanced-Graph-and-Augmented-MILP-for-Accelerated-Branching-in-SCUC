import os
import shutil
import pandas as pd
import numpy as np

def process_csv_with_random(input_file_path, output_file_path, random_values, columns):
    input_df = pd.read_csv(input_file_path)  
    num_rows = len(input_df)

    # 确保随机数数量与行数匹配
    if len(random_values) != num_rows:
        raise ValueError("随机数数量与 CSV 行数不匹配！")

    output_df = input_df.copy()  # 创建一个新的 DataFrame 用于输出

    for i in range(num_rows):
        for col in columns:
            curr_col = output_df.columns[col]  # 获取要修改的列名
            output_df[curr_col] = output_df[curr_col].astype('float64')
            if col < len(output_df.columns):  # 确保列索引有效
                output_df.at[i, output_df.columns[col]] *= random_values[i]  # 修改指定列

    output_df.to_csv(output_file_path, index=False)  # 保存修改后的文件


if __name__ == "__main__":
    BASE_FOLDER = r"./data/instances/case118/case118_1"  # 原始文件夹根目录
    caseName = r"case118"                       # 算例基础名称
    # 新建文件夹总数
    NUM_FOLDERS = 2599
    BEGIN_NO = 1
    RANDOM_SEED_A = 42   # 随机种子 A
    RANDOM_SEED_B = 84   # 随机种子 B
    RANDOM_SEED_C = 123  # 随机种子 C
    RANDOM_SEED_D = 2025 # 随机种子 D（新增：用于E列插值）
    RANDOM_SEED_E = 999  # 随机种子 E（新增：用于F列 init_state_time）
    RANDOM_SEED_F = 777  # 随机种子 F（新增：用于随机删除线路行）
    RANDOM_SEED_G = 888  # 随机种子 F（新增：用于随机删除机组行）
    A_RANGE = (0.925, 1.075)  # a 的取值范围
    B_RANGE = (0.95, 1.05)    # b 的取值范围
    C_RANGE = (0.9, 1.1)      # c 的取值范围
    D_RANGE = (0.0, 1.0)      # d 的取值范围：用于在[Pmin, Pmax]间插值
    E_RANGE = (0.01, 2.0)
    F_RANGE = [(10, 50)]
    G_RANGE = [(2, 54)]
    G_RANGE_BEGIN_FOLDER = 2500
    L_RANGE_BEGIN_FOLDER = 2500

    col_c_index = 2  # Pmax 所在列（例如：第3列）
    col_d_index = 3  # Pmin 所在列（例如：第4列）
    e_col_index = 4  # iniP 所在列（例如：第5列）
    f_col_index = 5  # init_state_time 所在列（例如：第6列）

    target_columns = [12, 13, 14, 27, 28, 29, 30]     # 冷、热、最小发电价格、各分段报价

    # 确保输出路径存在
    OUTPUT_BASE_PATH = os.path.dirname(BASE_FOLDER)  # 输出路径与 原始文件夹 同一级目录
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

    # 主循环：生成文件夹并处理文件
    for y in range(BEGIN_NO + 1, BEGIN_NO + NUM_FOLDERS + 1):
        new_folder_name = f"{caseName}_{y}"
        new_folder_path = os.path.join(OUTPUT_BASE_PATH, new_folder_name)

        # 创建新文件夹
        os.makedirs(new_folder_path, exist_ok=True)

        # 打印当前正在处理的序号
        print(f"正在处理文件夹: {new_folder_name}")

        # 复制原始文件夹中的所有文件到新文件夹
        for file_name in os.listdir(BASE_FOLDER):
            src_file_path = os.path.join(BASE_FOLDER, file_name)
            dst_file_path = os.path.join(new_folder_path, file_name)
            shutil.copy(src_file_path, dst_file_path)

        # 构建文件路径
        system_load_path = os.path.join(new_folder_path, f"3-{caseName}-systemload.csv")
        bus_load_path = os.path.join(new_folder_path, f"2-{caseName}-busload.csv")
        generator_data_path = os.path.join(new_folder_path, f"5-{caseName}-unitdata.csv")
        line_data_path = os.path.join(new_folder_path, f"4-{caseName}-linedata.csv")

        # 读取有效行数
        system_load_df = pd.read_csv(system_load_path)
        bus_load_df = pd.read_csv(bus_load_path)
        generator_data_df = pd.read_csv(generator_data_path)
        line_data_df = pd.read_csv(line_data_path)

        num_system_load_rows = len(system_load_df)
        num_bus_load_rows = len(bus_load_df)
        num_generator_data_rows = len(generator_data_df)
        num_line_data_rows = len(line_data_df)

        # 生成随机数
        np.random.seed(RANDOM_SEED_A + y)
        a_values = np.random.uniform(A_RANGE[0], A_RANGE[1], num_system_load_rows)

        np.random.seed(RANDOM_SEED_B + y)
        b_values = np.random.uniform(B_RANGE[0], B_RANGE[1], num_generator_data_rows)

        np.random.seed(RANDOM_SEED_C + y)
        c_values = np.random.uniform(C_RANGE[0], C_RANGE[1], num_bus_load_rows)

        np.random.seed(RANDOM_SEED_D + y)
        d_values = np.random.uniform(D_RANGE[0], D_RANGE[1], num_generator_data_rows)

        np.random.seed(RANDOM_SEED_E + y)
        e_values = np.random.uniform(E_RANGE[0], E_RANGE[1], num_generator_data_rows)

        np.random.seed(RANDOM_SEED_F + y)
        line_candidates = []
        for low, high in F_RANGE:
            line_candidates.extend(range(low, high + 1))        # range 是左闭右开，所以要 high+1
        line_to_delete = np.random.choice(line_candidates)

        np.random.seed(RANDOM_SEED_G + y)
        unit_candidates = []
        for low, high in G_RANGE:
            unit_candidates.extend(range(low, high + 1))
        unit_candidates = sorted(set(unit_candidates))
        k = min(np.random.randint(1, 4), len(unit_candidates))
        if k > 0:
            units_to_delete = np.random.choice(unit_candidates, size=k, replace=False)
        else:
            units_to_delete = np.array([])

        print(f"  生成的随机数数量: a_values={len(a_values)}, b_values={len(b_values)}, c_values={len(c_values)}, "
              f"d_values={len(d_values)}, e_values={len(e_values)}")

        # 处理 3-{caseName}-systemload.csv
        process_csv_with_random(system_load_path, system_load_path, a_values, [1, 2])
        updated_system_load_df = pd.read_csv(system_load_path)
        max_value_col_2 = updated_system_load_df.iloc[:, 1].max()
        updated_system_load_df.iloc[:, 3] = updated_system_load_df.iloc[:, 1] / max_value_col_2
        updated_system_load_df.to_csv(system_load_path, index=False)

        # 处理 5-{caseName}-unitdata.csv：先处理原目标列
        process_csv_with_random(generator_data_path, generator_data_path, b_values, target_columns)

        gen_df = pd.read_csv(generator_data_path)
        pmax_vals = gen_df.iloc[:, col_c_index]
        pmin_vals = gen_df.iloc[:, col_d_index]
        e_col_name = gen_df.columns[e_col_index]
        e_original = gen_df.iloc[:, e_col_index]

        e_new = np.where(
            e_original != 0,
            d_values * (pmax_vals - pmin_vals) + pmin_vals,
            0.0
        )

        e_col = gen_df.columns[e_col_index]  # 获取要修改的列名
        gen_df[e_col] = gen_df[e_col].astype('float64')
        gen_df.iloc[:, e_col_index] = e_new
        gen_df.to_csv(generator_data_path, index=False)

        gen_df = pd.read_csv(generator_data_path)  # 重新读取（确保包含前面的修改）
        f_original = gen_df.iloc[:, f_col_index]   # 原始 F 列值
        f_new = np.ceil(f_original * e_values).astype(int)     # 缩放后向上取整,强制转化为整数
        gen_df.iloc[:, f_col_index] = f_new

        if y >= G_RANGE_BEGIN_FOLDER:
            total_unit_df = len(gen_df)
            valid_units_to_delete = []

            for idx in units_to_delete:
                if 0 <= idx < total_unit_df:
                    valid_units_to_delete.append(idx)
                else:
                    print(f"  警告：机组索引 {idx} 超出范围 [0, {total_unit_df - 1}]，跳过")

            if valid_units_to_delete:
                gen_df = gen_df.drop(index=valid_units_to_delete).reset_index(drop=True)
                deleted_lines = [str(i + 1) for i in valid_units_to_delete]  # 转为1-based显示
                print(f"  已删除unitdata的第 {', '.join(deleted_lines)} 行（原始索引 {valid_units_to_delete}）")
            else:
                print("  无有效机组可删除")

        gen_df.to_csv(generator_data_path, index=False)

        line_df = pd.read_csv(line_data_path)
        total_lines = len(line_df)

        if y >= L_RANGE_BEGIN_FOLDER:
            if 0 <= line_to_delete < total_lines:
                row_to_check = line_df.iloc[line_to_delete]
                if len(line_df.columns) > 7 and isinstance(row_to_check.iloc[7], str) and row_to_check.iloc[7].strip().upper() == "YES":
                    print(f"  跳过删除第 {line_to_delete+1} 行（H列为 'YES'）")
                else:
                    line_df = line_df.drop(index=line_to_delete).reset_index(drop=True)
                    print(f"  已删除线路数据的第 {line_to_delete+1} 行（原始索引 {line_to_delete}）")
            else:
                print(f"  无法删除的线路索引: {line_to_delete}，跳过删除")

        line_df.to_csv(line_data_path, index=False)

        # 计算 c_values 的总和
        sum_c_values = np.sum(c_values)

        # 计算 max_demand 和 sum_demand 和 各母线的负荷 的原始占比
        max_demand = updated_system_load_df.iloc[:, 1].max()  # 第 2 列最大值
        sum_demand = bus_load_df.iloc[:, 1].sum()  # 第 2 列总和
        origin_rate = bus_load_df.iloc[:, 1] / sum_demand  # 第2列各元素占 原始systemload 的比例
        total_adjusted = sum(c * r for c, r in zip(c_values, origin_rate))

        # 缩放 2-{caseName}-busload.csv 的第二列
        bus_load_df.iloc[:, 1] *= (max_demand / sum_demand)
        bus_load_df.to_csv(bus_load_path, index=False)

        # 使用 c_values * origin_rate[i] / total_adjusted 作为后来的各busload占比，调整每行第二列
        for i in range(num_bus_load_rows):
            bus_load_df.at[i, bus_load_df.columns[1]] = \
                c_values[i] * origin_rate[i] * max_demand / total_adjusted

        bus_load_df.to_csv(bus_load_path, index=False)

    print("处理完成！")
