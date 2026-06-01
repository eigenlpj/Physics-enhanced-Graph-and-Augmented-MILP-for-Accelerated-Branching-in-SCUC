import subprocess
import os
import sys
from pathlib import Path

def main():
    root_dir = Path(__file__).parent
    os.chdir(root_dir)

    start = 1
    end = 1

    # case_name = "case118"
    case_name = "case1888"
    # case_name = "case2383"
    # case_name = "Anonymous"

    iter_finish_flag = 0
    for i in range(start, end + 1):
        print(f"{'='*50}")
        print(f"Current iter: {i}...")
        print(f"{'='*50}")

        # # === Step 1: Sample colletion ===
        # gen_njobs = 12
        # gen_train_size = 10000
        # gen_valid_size = 2000
        # gen_cmd = [
        #     sys.executable, "02_generate_dataset_case1888.py",
        #     case_name,
        #     "--njobs", str(gen_njobs),
        #     '--train_size', str(gen_train_size),
        #     '--valid_size', str(gen_valid_size),
        #     '--iter', str(i),
        # ]
        # try:
        #     result_gen_cmd = subprocess.run(gen_cmd, check=True, text=True)
        # except subprocess.CalledProcessError as e:
        #     print(f"Dataset generation failed: {e}")
        #     break

        # === Step 2 and 3: Data augmentation and PGNN training ===
        gcnn_train_batch_size = 8
        gcnn_valid_batch_size = 16
        current_lr = 1e-3
        epoch = 100
        current_augs = 1
        print(f"-------------------------------")
        print(f"PGNN train begin: iter={i}, epoch={epoch}")
        gcnn_data_dir = "D:/LiJiamigFile/CAMBranch-ljmdata/KIDA_data/samples"
        trainGCNN_cmd = [
            sys.executable, "03_train_gcnn_pyG_aug_generate.py",
            "--case", case_name,
            '--iter', str(i),
            "--gpu", "0",
            "--data_dir", gcnn_data_dir,
            "--accum-iter", "4",
            "--alpha_cl", "0.05",
            "--alpha_reg", "0.01",
            "--train_batch_size", str(gcnn_train_batch_size),
            "--valid_batch_size", str(gcnn_valid_batch_size),
            "--lr", str(current_lr),
            "--num_augs", str(current_augs),
            "--s1", "29.0",
            "--s2", "29.0",
            "--rel_con", "0.1",
            "--shift_mode", "abs_shift",
            # "--shift_mode", "rel_shift",
            # "--train_mode", "gcnn_aug",
            "--train_mode", "gcnn_aug_with_huake",
            # "--if_pgraph", "NO",
            # "--if_AMILP", "NO",
            "--max_epochs", str(epoch),
        ]
        try:
            result_trainGCNN_cmd = subprocess.run(trainGCNN_cmd, check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"GCNN train fail: {e}")
            break  # 可选：失败时停止

        if i == end:
            iter_finish_flag = 1
        print(f"\titer: {i} is completed.\n")

    if iter_finish_flag ==1:
        print("All iterations have been successfully completed!")

if __name__ == "__main__":
    main()
