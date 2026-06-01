"""
Adapted from ds4dm/learn2branch (https://github.com/ds4dm/learn2branch/blob/master/02_generate_dataset.py) and CAMbranch (https://github.com/linjc16)
Modified for PGNN (Physics-informed graph network) branching framework under the same MIT License.
"""
import os
import sys
import argparse
import pathlib
import numpy as np
import pickle
import pdb
import gzip
from collections import defaultdict
import math
import torch
import torch.nn.functional as F
import torch_geometric
from sympy import false
from torch.utils.data import DataLoader
import torch.nn as nn
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import GradScaler
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import random

sys.path.insert(0, os.path.abspath(f'models'))
from models.model import GNNPolicy
from utils_aug10 import log, pad_tensor, AugmentedGraphDataset_LJM_v0, collate_fn_aug

def compute_huake_loss(logits, soft_labels, n_can_sample, labels_true_index, depth_LJM, device):
    """
    logits: [n_total_vars]
    soft_labels: [n_total_candidates]
    n_can_sample: [B]
    labels_true_index: [n_total_candidates]
    depth_LJM: [B]
    device: cpu/gpu
    """
    selected_logits = torch.gather(logits, dim=0, index=labels_true_index)  # [M]

    selected_labels = soft_labels

    start_idx = 0
    losses = []

    for i, n in enumerate(n_can_sample):
        n = n.item()
        end_idx = start_idx + n

        logit_part = selected_logits[start_idx:end_idx]     # [n_i]
        label_part = selected_labels[start_idx:end_idx]     # [n_i]

        log_probs = F.log_softmax(logit_part, dim=0)        # [n_i]

        loss_sample = - (label_part * log_probs).sum()

        depth_single = depth_LJM[i] / 10
        C = (1 + math.exp(-0.5))
        w = C * torch.sigmoid(torch.tensor(0.5, device=device) - depth_single).detach()
        loss_sample = loss_sample * w

        losses.append(loss_sample)
        start_idx = end_idx
    return torch.stack(losses)  # [B]


def get_cosine_loss(anchor, positive):
    anchor = anchor / torch.norm(anchor, dim=-1, keepdim=True)         # shape [B, D]
    positive = positive / torch.norm(positive, dim=-1, keepdim=True)   # shape [B, D]
    logits = torch.matmul(anchor, positive.T)                          # shape [B, B],
    # When the loss value is small, each main diagonal element of the logits is almost equal to 1,
    # and the remaining elements are almost equal to 0, indicating that the contrastive learning has converged.
    logits = logits - torch.max(logits, 1, keepdim=True)[0].detach()
    targets = torch.arange(logits.shape[1]).long().to(logits.device)
    loss = nn.CrossEntropyLoss(reduction='mean')(logits, targets)
    
    return loss


def pretrain(policy, pretrain_loader):
    policy.pre_train_init()
    i = 0
    while True:
        for batch in pretrain_loader:
            batch.to(device)
            if not policy.pre_train(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features):
                break

        if policy.pre_train_next() is None:
            break
        i += 1
    return i


def process(args, policy, data_loader, top_k=[1, 3, 5, 10], train_mode='gcnn_aug', optimizer=None, scaler=None):
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))
    mean_entropy = 0

    if optimizer is not None:
        policy.train()
    else:
        policy.eval()
    batch_index = -1
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(data_loader):
            if optimizer is not None:
                batch, batch_aug = batch
                batch_index += 1
                batch, batch_aug = batch.to(device), batch_aug.to(device)
                logits_orig, graph_embeds = policy.forward_with_graph_embs(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features, batch.nb_constraints, batch.nb_variables)
                logits_aug_orig, graph_embeds_aug = policy.forward_with_graph_embs(batch_aug.constraint_features, batch_aug.edge_index, batch_aug.edge_attr, batch_aug.variable_features, batch_aug.nb_constraints, batch_aug.nb_variables)
                logits = pad_tensor(logits_orig[batch.candidates], batch.nb_candidates)
                logits_aug = pad_tensor(logits_aug_orig[batch_aug.candidates], batch_aug.nb_candidates)
                if train_mode == 'gcnn_aug':
                    cross_entropy_loss = F.cross_entropy(logits, batch.candidate_choices, reduction='mean')
                    if args.if_AMILP == 'YES':
                        cross_entropy_loss += F.cross_entropy(logits_aug, batch_aug.candidate_choices, reduction='mean')
                        cross_entropy_loss /= 2
                elif train_mode == 'gcnn_aug_with_huake':
                    loss_per_sample_1 = compute_huake_loss(
                        logits=logits_orig,  # [n_total_vars]
                        soft_labels=batch.z_i_with_w,  # [n_total_candidates, 1]
                        n_can_sample=batch.nb_candidates,  # [B]
                        labels_true_index=batch.candidates,  # [n_total_candidates]
                        depth_LJM=batch.depth_i,  # [B]
                        device=device
                    )  #  [B]
                    cross_entropy_loss_1 = loss_per_sample_1.mean()
                    loss_per_sample_2 = compute_huake_loss(
                        logits=logits_aug_orig,  # [n_total_vars]
                        soft_labels=batch.z_i_with_w,  # [n_total_candidates, 1]
                        n_can_sample=batch.nb_candidates,  # [B]
                        labels_true_index=batch.candidates,  # [n_total_candidates]
                        depth_LJM=batch.depth_i,  # [B]
                        device=device
                    )  #  [B]
                    cross_entropy_loss_2 = loss_per_sample_2.mean()
                    if args.if_AMILP == 'YES':
                        cross_entropy_loss = (cross_entropy_loss_1 + cross_entropy_loss_2) / 2
                    else:
                        cross_entropy_loss = cross_entropy_loss_1


                reg_loss_fn = nn.MSELoss()
                reg_loss = reg_loss_fn(F.logsigmoid(logits), F.logsigmoid(logits_aug))

                cl_loss = get_cosine_loss(graph_embeds, graph_embeds_aug)

                if args.if_AMILP == 'YES':
                    loss = cross_entropy_loss + args.alpha_cl * cl_loss + args.alpha_reg * reg_loss
                else:
                    loss = cross_entropy_loss

            else:

                batch = batch.to(device)
                logits_orig = policy(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
                logits = pad_tensor(logits_orig[batch.candidates], batch.nb_candidates)

                if train_mode == 'gcnn_aug':
                    cross_entropy_loss = F.cross_entropy(logits, batch.candidate_choices, reduction='mean')
                    loss = cross_entropy_loss
                elif train_mode == 'gcnn_aug_with_huake':
                    loss_per_sample_1 = compute_huake_loss(
                        logits=logits_orig,  # [n_total_vars]
                        soft_labels=batch.z_i_with_w,  # [n_total_candidates, 1]
                        n_can_sample=batch.nb_candidates,  # [B]
                        labels_true_index=batch.candidates,  # [n_total_candidates]
                        depth_LJM=batch.depth_i,  # [B]
                        device=device
                    )  #  [B]
                    cross_entropy_loss = loss_per_sample_1.mean()
                    loss = cross_entropy_loss

            loss = loss / args.accum_iter

            if scaler is not None:
                scaler.scale(loss).backward()

            if optimizer is not None:
                if scaler is not None:
                    if ((step + 1) %  args.accum_iter == 0) or ((step + 1) == len(data_loader)):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            kacc = []
            for k in top_k:
                if logits.size()[-1] < k:
                    kacc.append(1.0)
                    continue
                pred_top_k = logits.topk(k).indices
                pred_top_k_true_scores = true_scores.gather(-1, pred_top_k)
                accuracy = (pred_top_k_true_scores == true_bestscore).any(dim=-1).float().mean().item()
                kacc.append(accuracy)
            kacc = np.asarray(kacc)
            mean_loss += cross_entropy_loss.item() * batch.num_graphs
            mean_kacc += kacc * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed
    
    return mean_loss, mean_kacc


def find_samples(case_name:str, i_max:int, base_dir:str, folder_name:str, user_percentage: float=1.0, log_dir: str=None):
    folder_me = []
    iter1_sample_num = 0
    iter2plus_sample_num = 0
    iter2plus_selected_sample_num = 0
    train_files = []

    instance_groups = defaultdict(list)
    folder_path = pathlib.Path(base_dir) / f"{case_name}_iter{i_max}" / folder_name
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


if __name__ == "__main__":

    mp.set_start_method('spawn')
    rng = np.random.default_rng(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--case',
        type=str, default='Blank Name', help='Please get me your caseName without number, e.g. case118 rather than case118_1001',
    )
    parser.add_argument(
        '--iter',
        help='Number of current iteration.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument('--model-name', type=str, default='gcnn_aug')
    parser.add_argument('--data_dir', type=str, default='Blank path',
                        help='Please get me your upper-level path of train and valid data ')
    parser.add_argument('--accum-iter', type=int, default=2)
    parser.add_argument('--alpha_cl', type=float, default=0.05)
    parser.add_argument('--alpha_reg', type=float, default=0.01)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--valid_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_augs', type=int, default=1)
    parser.add_argument('--s1', type=float, default=4.0, help='Please get me your multiplier_dis')
    parser.add_argument('--s2', type=float, default=20.0,help='Please get me your multiplier_con')
    parser.add_argument('--rel_con', type=float, help='Please get me your rel_con if your shift_mode is rel_shift')
    parser.add_argument(
        '--shift_mode',
        type=str,
        help='Which mode do you use for the shift of your continuous variable?',
        choices=['abs_shift', 'rel_shift'],
    )
    parser.add_argument(
        '--if_pgraph',
        type=str,
        help='Do you use Pgraph?',
        choices=['YES', 'NO'],
        default='YES',
    )
    parser.add_argument(
        '--if_AMILP',
        type=str,
        help='Do you use AMILP?',
        choices=['YES', 'NO'],
        default='YES',
    )
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--second_filer_pct', type=float, default=1.0)
    parser.add_argument(
        '--train_mode',
        type=str,
        help='Do you need to train gcnn_aug_with_huake or gcnn_aug?',
        choices=['gcnn_aug_with_huake', 'gcnn_aug'],
        default='gcnn_aug',
    )

    args = parser.parse_args()
    max_epochs = args.max_epochs
    batch_size = args.train_batch_size
    pretrain_batch_size = args.train_batch_size
    valid_batch_size = args.valid_batch_size
    lr = args.lr
    top_k = [1, 3, 5, 10]

    running_dir = f"D:/LiJiamigFile/CAMBranch-ljmdata/KIDA_log/{args.case}/{args.model_name}/{args.seed}/{args.alpha_cl}_{args.alpha_reg}"
    os.makedirs(running_dir, exist_ok=True)
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
        device_scaler = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:{args.gpu}"
        device_scaler = "cuda"

    largest_x = -1000
    gcnn_checkpoint_paths = []
    for filename in os.listdir(running_dir):
        if filename.startswith("gcnn_aug_") and filename.endswith(".pkl"):
            x = int(filename.split("_")[-1].split(".")[0])
            gcnn_checkpoint_paths.append((os.path.join(running_dir, filename), x))
    gcnn_checkpoint_paths.sort(key=lambda x: x[1])
    if len(gcnn_checkpoint_paths) >= 1:
        largest_x = gcnn_checkpoint_paths[-1][1]
        new_x = largest_x + 1
    else:
        new_x = 1
    new_x = args.iter
    pklfile = os.path.join(running_dir, f'{args.model_name}_{new_x}.pkl')
    if os.path.exists(pklfile):
        print(f"The file gcnn_aug_{new_x}.pkl already exists in the path {running_dir}. Please delete it manually or make a backup before training the model...")
        exit(1)

    torch.manual_seed(args.seed)

    policy = GNNPolicy(cl=True, if_pgraph=args.if_pgraph).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scaler = GradScaler(device_scaler)
    best_acc1 = -1.0
    num_bad_epoch = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.2)

    train_folder_name, train_files = find_samples(args.case, args.iter, args.data_dir, "train_MC_UE",
                                                  user_percentage=args.second_filer_pct,log_dir=running_dir)
    print(f"-------------------------------")
    print(f"Train_folder_name:")
    for folder_i in train_folder_name:
        print(f"{folder_i}")
    print(f"-------------------------------")
    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]
    valid_folder_name, valid_files = find_samples(case_name=args.case, i_max=1, base_dir=args.data_dir, folder_name="valid",
                                                  user_percentage=args.second_filer_pct,log_dir=running_dir)
    print(f"-------------------------------")
    print(f"Valid_folder_name:")
    for folder_i in valid_folder_name:
        print(f"{folder_i}")
    print(f"-------------------------------")

    if len(train_files) > batch_size:
        epoch_size = int(len(train_files)/batch_size)
    else:
        epoch_size = 1
        batch_size = len(train_files)
        pretrain_batch_size = len(train_files)

    ### LOG ###
    logfile = os.path.join(running_dir, f'{args.model_name}.txt')
    log(f"case:           {args.case}", logfile)
    log(f"iter:           {new_x}", logfile)
    log(f"model save dir: {running_dir}", logfile)
    log(f"train data dir: ", logfile)
    for folder_i in train_folder_name:
        log(f"\t\t\t\t {folder_i}", logfile)
    log(f"valid data dir: ", logfile)
    for folder_i in valid_folder_name:
        log(f"\t\t\t\t {folder_i}", logfile)
    log(f"pretrain samples num: {len(pretrain_files)}", logfile)
    log(f"train samples num:    {len(train_files)}", logfile)
    log(f"valid samples num:    {len(valid_files)}", logfile)
    log(f"max_epochs:           {max_epochs}", logfile)
    log(f"batch_size:           {batch_size}", logfile)
    log(f"pretrain_batch_size:  {pretrain_batch_size}", logfile)
    log(f"valid_batch_size :    {valid_batch_size }", logfile)
    log(f"lr: {lr}", logfile)
    log(f"accum_iter: {args.accum_iter}", logfile)
    log(f"alpha_cl: {args.alpha_cl}", logfile)
    log(f"alpha_reg: {args.alpha_reg}", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"case: {args.case}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed: {args.seed}", logfile)
    log(f"num_augs: {args.num_augs}", logfile)
    log(f"s1: {args.s1}", logfile)
    log(f"s2: {args.s2}", logfile)
    log(f"rel_con: {args.rel_con}", logfile)
    log(f"shift_mode: {args.shift_mode}", logfile)
    log(f"train_mode: {args.train_mode}", logfile)
    log(f"if_pgraph: {args.if_pgraph}", logfile)
    log(f"if_AMILP: {args.if_AMILP}", logfile)

    pretrain_data = AugmentedGraphDataset_LJM_v0(pretrain_files, num_augs=0, if_pgraph=args.if_pgraph)
    pretrain_loader = torch_geometric.loader.DataLoader(pretrain_data, pretrain_batch_size, shuffle=False, num_workers=4)
    valid_data = AugmentedGraphDataset_LJM_v0(valid_files, num_augs=0, if_pgraph=args.if_pgraph)
    valid_loader = torch_geometric.loader.DataLoader(valid_data, valid_batch_size, shuffle=False, num_workers=4)
    epoch_train_files = rng.choice(train_files, epoch_size * batch_size, replace=false)

    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)
        if epoch == 0:
            if new_x == 1:
                n = pretrain(policy, pretrain_loader)
                log(f"PRETRAINED {n} LAYERS", logfile)
        else:
            train_data = AugmentedGraphDataset_LJM_v0(epoch_train_files, num_augs=args.num_augs,
                                                      multiplier_dis=args.s1, multiplier_con=args.s2,
                                                      rel_con=args.rel_con, shift_mode=args.shift_mode, if_pgraph=args.if_pgraph)
            train_loader = DataLoader(train_data, batch_size, shuffle=True, collate_fn=collate_fn_aug, num_workers=4)
            if epoch == 1:
                log(f"Train_loader is set at every epoch, num_train_batch: {len(train_loader)} ", logfile)
            train_loss, train_kacc = process(args, policy, train_loader, top_k, args.train_mode, optimizer, scaler)
            log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)
        
        # Validation
        valid_loss, valid_kacc = process(args, policy, valid_loader, top_k, args.train_mode, optimizer=None)
        log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

        scheduler.step(valid_kacc[0])
        num_bad_epoch += 1
        if valid_kacc[0] > best_acc1:
            best_acc1 = valid_kacc[0]
            num_bad_epoch = 0
            torch.save(policy.state_dict(), pathlib.Path(running_dir)/f'{args.model_name}_{new_x}.pkl')
            log(f"  best model so far", logfile)
        elif num_bad_epoch == 10:
            log(f"  10 epochs without improvement, decreasing learning rate, current lr: {scheduler.get_last_lr()[0]:.2e}", logfile)
        elif num_bad_epoch == 20:
            log(f"  20 epochs without improvement, early stopping, current lr: {scheduler.get_last_lr()[0]:.2e}", logfile)
            break

    policy.load_state_dict(torch.load(pathlib.Path(running_dir)/f'{args.model_name}_{new_x}.pkl', map_location='cpu', weights_only=True))
    valid_loss, valid_kacc = process(args, policy, valid_loader, top_k, args.train_mode, optimizer=None)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

    # Test
    # test_files = [str(file) for file in (pathlib.Path(f'{args.root_dir}/data/samples')/problem_folder/'test').glob('sample_*.pkl')]
    #
    # test_data = AugmentedGraphDataset(test_files, num_augs=0)
    # # test_loader = torch_geometric.loader.DataLoader(test_data, valid_batch_size, shuffle=False, num_workers=4)
    #
    # test_loss, test_kacc = process(args, policy, test_loader, top_k, args.train_mode, optimizer=None)
    # log(f"TEST LOSS: {test_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.5f}" for k, acc in zip(top_k, test_kacc)]), logfile)
