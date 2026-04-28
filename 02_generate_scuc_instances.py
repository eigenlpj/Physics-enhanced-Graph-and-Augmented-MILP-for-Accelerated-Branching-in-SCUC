import os
import subprocess
from multiprocessing import Pool


BASE_DIR = r"./data/instances/case118/case118_1"
caseName = r"case118"
slackBus = r"66"

# Cleanup switch (True: delete old .lp files and PTDF_matrix.csv; False: keep existing PTDF_matrix.csv)
CLEAN_UP = False
# CLEAN_UP = True

SCRIPT_PATH = r"./IEEE_g.py"
Begin_no = 1
NUM_FOLDERS = 2600
NUM_CORES = 6

def clean_directory(base_dir):
    """
    遍历 base_dir 下的所有子文件夹，删除 .lp 文件和 PTDF_matrix.csv
    """
    print(f"\n  正在扫描目录: {base_dir}")
    deleted_count = 0

    # 遍历根目录下的所有文件夹
    for root_dir, dirs, files in os.walk(base_dir):
        # 检查当前层级是否存在目标文件
        files_to_delete = []

        # 查找 .lp 文件
        for f in files:
            if f.endswith('.lp') or f == 'PTDF_matrix.csv':
                file_path = os.path.join(root_dir, f)
                files_to_delete.append(file_path)

        # 执行删除
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                # print(f"   - 已删除: {file_path}") # 如果不需要详细列表，可以注释掉这行
                deleted_count += 1
            except Exception as e:
                print(f"   - 删除失败 {file_path}: {e}")

    if deleted_count > 0:
        print(f" 清理完成！共删除 {deleted_count} 个文件 (.lp 和 PTDF_matrix.csv)。")
    else:
        print("  未找到需要删除的文件。")
    print("-" * 30)


def run_script(folder_name):
    """
    运行 IEEE.py 脚本，并传递对应的文件夹路径作为参数。
    :param folder_name: 当前处理的文件夹名称（如 'case118' 或 'case118-1'）
    """
    dir_path = os.path.join(BASE_DIR, folder_name)  # 构造完整路径
    print(f"正在运行脚本，处理文件夹: {dir_path}")

    # 使用 subprocess 启动 IEEE.py 脚本，并传递 建模程序文件路径、数据文件夹路径、算例基础名称(case118、case300、case2383...)、平衡节点号 一共4个参数
    command = ["python", SCRIPT_PATH, dir_path, caseName, slackBus]
    try:
        result = subprocess.run(command, check=True)
        print(f"完成处理文件夹: {dir_path}, 算例基础名称: {caseName}, 平衡节点号: {slackBus}")
    except subprocess.CalledProcessError as e:
        print(f"处理文件夹 {dir_path} 时出错: {e}")


if __name__ == "__main__":
    if CLEAN_UP:
        print(">>> 检测到 CLEAN_UP = True，正在执行清理旧的.lp文件和PTDF_matrix.csv任务...")
        clean_directory(BASE_DIR)
    else:
        print(">>> 检测到 CLEAN_UP = False，跳过清理步骤。")


    # 获取所有需要处理的文件夹名称
    folder_names = [f"{caseName}_{y}" for y in range(Begin_no, Begin_no + NUM_FOLDERS)]

    # 使用多进程池并行运行，核心数固定为 12
    print(f"使用 {NUM_CORES} 个核心并行运行...")

    with Pool(processes=NUM_CORES) as pool:
        pool.map(run_script, folder_names)

    print("所有任务已完成！")
