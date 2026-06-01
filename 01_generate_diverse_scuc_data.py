import os
import shutil
import pandas as pd
import numpy as np

def process_csv_with_random(input_file_path, output_file_path, random_values, columns):
    """
    Process the CSV file and save the modified file. Each row uses a different random number.
    :param input_file_path
    :param output_file_path
    :param random_values
    :param columns: Indexes of the columns to be processed (starting from 0)
    """
    input_df = pd.read_csv(input_file_path)
    num_rows = len(input_df)

    if len(random_values) != num_rows:
        raise ValueError("The number of random numbers does not match the number of rows in the CSV file!")

    output_df = input_df.copy()

    for i in range(num_rows):
        for col in columns:
            curr_col = output_df.columns[col]
            output_df[curr_col] = output_df[curr_col].astype('float64')
            if col < len(output_df.columns):
                output_df.at[i, output_df.columns[col]] *= random_values[i]

    output_df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    BASE_FOLDER = r".../data/instances/case118/case118_1"  # Root directory of the original folder
    caseName = r"case118"  # Case base Name
    NUM_FOLDERS = 2599
    BEGIN_NO = 1
    RANDOM_SEED_A = 42
    RANDOM_SEED_B = 84
    RANDOM_SEED_C = 123
    RANDOM_SEED_D = 2025
    RANDOM_SEED_E = 999
    RANDOM_SEED_F = 777
    RANDOM_SEED_G = 888
    A_RANGE = (0.925, 1.075)
    B_RANGE = (0.95, 1.05)
    C_RANGE = (0.9, 1.1)
    D_RANGE = (0.0, 1.0)      # Used for interpolation within the range of [Pmin, Pmax]
    E_RANGE = (0.01, 2.0)     # Used for scaling of the "iniTime" column in F
    F_RANGE = [(10, 50)]                                      # N-1 for case118
    # F_RANGE = [(25, 30), (33, 50), (57, 62), (105, 130)]    # N-1 for Anonymous
    # F_RANGE = [(5, 100)]                                    # N-1 for Polish 2383
    # F_RANGE = [(30,35), (38, 42), (46, 60)]                 # N-1 for French 1888
    G_RANGE = [(2, 54)]                                       # Unit outages for case118
    # G_RANGE = [(7, 41)]                                     # Unit outages for Anonymous
    # G_RANGE = [(2, 323)]                                    # Unit outages for Polish 2383
    # G_RANGE = [(56, 296)]                                   # Unit outages for French 1888
    G_RANGE_BEGIN_FOLDER = 2500                               # N-1 begin number
    L_RANGE_BEGIN_FOLDER = 2500                               # Unit outage begin number

    col_c_index = 2  # Pmax column
    col_d_index = 3  # Pmin column
    e_col_index = 4  # iniP column
    f_col_index = 5  # init_state_time column
    target_columns = [12, 13, 14, 27, 28, 29, 30]                             # bid columns in case118/Polish 2383/French 1888
    # target_columns = [12, 13, 14, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]   # bid columns in Anonymous

    OUTPUT_BASE_PATH = os.path.dirname(BASE_FOLDER)
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

    for y in range(BEGIN_NO + 1, BEGIN_NO + NUM_FOLDERS + 1):
        new_folder_name = f"{caseName}_{y}"
        new_folder_path = os.path.join(OUTPUT_BASE_PATH, new_folder_name)

        os.makedirs(new_folder_path, exist_ok=True)

        print(f"Currently processing folder: {new_folder_name}")

        for file_name in os.listdir(BASE_FOLDER):
            src_file_path = os.path.join(BASE_FOLDER, file_name)
            dst_file_path = os.path.join(new_folder_path, file_name)
            shutil.copy(src_file_path, dst_file_path)

        system_load_path = os.path.join(new_folder_path, f"3-{caseName}-systemload.csv")
        bus_load_path = os.path.join(new_folder_path, f"2-{caseName}-busload.csv")
        generator_data_path = os.path.join(new_folder_path, f"5-{caseName}-unitdata.csv")
        line_data_path = os.path.join(new_folder_path, f"4-{caseName}-linedata.csv")

        system_load_df = pd.read_csv(system_load_path)
        bus_load_df = pd.read_csv(bus_load_path)
        generator_data_df = pd.read_csv(generator_data_path)
        line_data_df = pd.read_csv(line_data_path)

        num_system_load_rows = len(system_load_df)
        num_bus_load_rows = len(bus_load_df)
        num_generator_data_rows = len(generator_data_df)
        num_line_data_rows = len(line_data_df)

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
            line_candidates.extend(range(low, high + 1))        # The range is left-closed and right-open, so we need to use high + 1.
        line_to_delete = np.random.choice(line_candidates)      # random
        # if y - 1 >= len(line_candidates):
        #     z = len(line_candidates) - 1
        # else:
        #     z = y - 2
        # line_to_delete = line_candidates[z]                   # Set a fixed value

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

        print(f"  The number of generated random numbers: a_values={len(a_values)}, b_values={len(b_values)}, c_values={len(c_values)}, "
              f"d_values={len(d_values)}, e_values={len(e_values)}")

        process_csv_with_random(system_load_path, system_load_path, a_values, [1, 2])
        updated_system_load_df = pd.read_csv(system_load_path)
        max_value_col_2 = updated_system_load_df.iloc[:, 1].max()
        updated_system_load_df.iloc[:, 3] = updated_system_load_df.iloc[:, 1] / max_value_col_2
        updated_system_load_df.to_csv(system_load_path, index=False)

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

        e_col = gen_df.columns[e_col_index]
        gen_df[e_col] = gen_df[e_col].astype('float64')
        gen_df.iloc[:, e_col_index] = e_new
        gen_df.to_csv(generator_data_path, index=False)

        gen_df = pd.read_csv(generator_data_path)
        f_original = gen_df.iloc[:, f_col_index]
        f_new = np.ceil(f_original * e_values).astype(int)
        gen_df.iloc[:, f_col_index] = f_new

        if y >= G_RANGE_BEGIN_FOLDER:
            total_unit_df = len(gen_df)
            valid_units_to_delete = []

            for idx in units_to_delete:
                if 0 <= idx < total_unit_df:
                    valid_units_to_delete.append(idx)
                else:
                    print(f"  Warning: The unit index {idx} is out of range [0, {total_unit_df - 1}], skipping.")

            if valid_units_to_delete:
                gen_df = gen_df.drop(index=valid_units_to_delete).reset_index(drop=True)
                deleted_lines = [str(i + 1) for i in valid_units_to_delete]
                print(f"  The {', '.join(deleted_lines)} lines of the deleted unit data (with the original index {valid_units_to_delete})")
            else:
                print("  No valid units available for deletion.")

        gen_df.to_csv(generator_data_path, index=False)

        line_df = pd.read_csv(line_data_path)
        total_lines = len(line_df)

        if y >= L_RANGE_BEGIN_FOLDER:
            # Check whether the "line_to_delete" is within the valid range
            if 0 <= line_to_delete < total_lines:
                row_to_check = line_df.iloc[line_to_delete]
                if len(line_df.columns) > 7 and isinstance(row_to_check.iloc[7], str) and row_to_check.iloc[7].strip().upper() == "YES":
                    print(f"  Skip the {line_to_delete + 1}th line (where the H column is 'YES').")
                else:
                    line_df = line_df.drop(index=line_to_delete).reset_index(drop=True)
                    print(f"  The {line_to_delete + 1}th line of the deleted line data (original index {line_to_delete})")
            else:
                print(f"  Cannot delete the following line index: {line_to_delete}, skipping")

        line_df.to_csv(line_data_path, index=False)

        sum_c_values = np.sum(c_values)

        max_demand = updated_system_load_df.iloc[:, 1].max()
        sum_demand = bus_load_df.iloc[:, 1].sum()
        origin_rate = bus_load_df.iloc[:, 1] / sum_demand
        total_adjusted = sum(c * r for c, r in zip(c_values, origin_rate))

        bus_load_df.iloc[:, 1] *= (max_demand / sum_demand)
        bus_load_df.to_csv(bus_load_path, index=False)

        for i in range(num_bus_load_rows):
            bus_load_df.at[i, bus_load_df.columns[1]] = \
                c_values[i] * origin_rate[i] * max_demand / total_adjusted

        bus_load_df.to_csv(bus_load_path, index=False)

    print("Processing completed!")
