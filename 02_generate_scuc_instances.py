import os
import subprocess
from multiprocessing import Pool

from sympy import false

# BASE_DIR = r"....\data\instances\case118"
# caseName = r"case118"
# slackBus = r"66"

BASE_DIR = r"....\data\instances\Anonymous"
caseName = r"Anonymous"
slackBus = r"13"

# BASE_DIR = r"....\data\data\instances\case1888"
# caseName = r"case1888"
# slackBus = r"1671"

# # BASE_DIR = r"....\data\instances\case2383"
# caseName = r"case2383"
# slackBus = r"18"


# Step 0. Clean the switch (True: Delete the old.lp file and PTDF_matrix.csv file; False: Keep the old files, that is, retain the old PTDF_matrix.csv file)
CLEAN_UP = False
# CLEAN_UP = True

# Step 1. Generate.lp file
SCRIPT_PATH = r"IEEE_g_Anonymous.py"      # Anonymous
# SCRIPT_PATH = r"IEEE_g.py"              # Others

Begin_no = 1
NUM_FOLDERS = 2600
NUM_CORES = 1  # Number of CPU cores used

def clean_directory(base_dir):
    print(f"\n  Scanning the directory: {base_dir}")
    deleted_count = 0

    for root_dir, dirs, files in os.walk(base_dir):
        files_to_delete = []

        for f in files:
            if f.endswith('.lp') or f == 'PTDF_matrix.csv':
                file_path = os.path.join(root_dir, f)
                files_to_delete.append(file_path)
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"   - Deletion failed {file_path}: {e}")

    if deleted_count > 0:
        print(f" Cleanup completed! A total of {deleted_count} files (.lp and PTDF_matrix.csv) were deleted.")
    else:
        print("  No file to be deleted was found.")
    print("-" * 30)


def run_script(folder_name):
    dir_path = os.path.join(BASE_DIR, folder_name)
    print(f"The script is running and processing the folder: {dir_path}")

    command = ["python", SCRIPT_PATH, dir_path, caseName, slackBus]
    try:
        result = subprocess.run(command, check=True)
        print(f"Completed processing folder: {dir_path}, System base name: {caseName}, Balance node number: {slackBus}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while processing the folder {dir_path}: {e}")


if __name__ == "__main__":
    if CLEAN_UP:
        print(">>> It has been detected that CLEAN_UP = True. The task of deleting the old.lp files and PTDF_matrix.csv is being executed...")
        clean_directory(BASE_DIR)
    else:
        print("It was detected that CLEAN_UP = False, so the cleanup steps were skipped.")
    folder_names = [f"{caseName}_{y}" for y in range(Begin_no, Begin_no + NUM_FOLDERS)]
    print(f"Run in parallel using {NUM_CORES} cores...")

    with Pool(processes=NUM_CORES) as pool:
        pool.map(run_script, folder_names)

    print("All tasks have been completed!")
