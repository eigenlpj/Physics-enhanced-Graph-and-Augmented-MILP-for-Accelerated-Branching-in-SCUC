"""
Adapted from ds4dm/learn2branch (https://github.com/ds4dm/learn2branch/blob/master/02_generate_dataset.py) and CAMbranch (https://github.com/linjc16)
Modified for PGNN (Physics-informed graph network) branching framework under the same MIT License.
"""
import os
import sys
import argparse
import pathlib
import numpy as np
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau


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


def process(policy, data_loader, top_k=[1, 3, 5, 10], train_mode='gasse', optimizer=None):
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))
    mean_entropy = 0
    
    if optimizer is not None:
        policy.train()
    else:
        policy.eval()
    
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(device)
            logits_orig = policy(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
            logits = pad_tensor(logits_orig[batch.candidates], batch.nb_candidates)  # (N, C_max), N: Number of samples in the batch, C_max: Maximum number of candidate variables contained in the sample

            if train_mode == 'gasse':
                cross_entropy_loss = F.cross_entropy(logits, batch.candidate_choices, reduction='mean')
                entropy = (-F.softmax(logits, dim=-1)*F.log_softmax(logits, dim=-1)).sum(-1).mean()
                loss = cross_entropy_loss - entropy_bonus*entropy

            elif train_mode == 'huake':
                loss_per_sample = compute_huake_loss(
                    logits=logits_orig,                    # [n_total_vars]
                    soft_labels=batch.z_i_with_w,          # [n_total_candidates, 1]
                    n_can_sample=batch.nb_candidates,      # [B]
                    labels_true_index=batch.candidates,    # [n_total_candidates]
                    depth_LJM=batch.depth_i,               # [B]
                    device=device
                )  # [B]
                cross_entropy_loss = loss_per_sample.mean()
                entropy = (-F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
                loss = cross_entropy_loss - entropy_bonus * entropy

            if optimizer is not None:
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
            mean_entropy += entropy.item() * batch.num_graphs
            mean_kacc += kacc * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed
    mean_entropy /= n_samples_processed
    return mean_loss, mean_kacc, mean_entropy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset', 'case118', 'case1888', '24GX', 'case2383'],
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
    parser.add_argument(
        '--model-name',
        type=str,
        default='gcnn_0.1'
    )
    parser.add_argument(
        '--root-dir',
        type=str,
        default='D:/LiJiamigFile/CAMBranch-ljmdata',
    )
    parser.add_argument(
        '--train_mode',
        type=str,
        help='Do you need to train huake or gasse?',
        choices=['huake', 'gasse'],
        default='huake',
    )
    
    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    max_epochs = 500
    batch_size = 16
    pretrain_batch_size = 16
    # valid_batch_size = 32
    valid_batch_size = 16
    lr = 0.001
    entropy_bonus = 0.0
    top_k = [1, 3, 5, 10]

    problem_folders = {
        'case118': 'case118_iter1',
        'case1888': 'case1888_iter1',
        '24GX': '24GX_iter1',
        'case2383': 'case2383_iter1',

    }
    problem_folder = problem_folders[args.problem]
    running_dir = f"D:/LiJiamigFile/CAMBranch-ljmdata/KIDA_log/{args.problem}/{args.model_name}_{args.train_mode}/{args.seed}"
    os.makedirs(running_dir, exist_ok=True)

    ### PYTORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"
    import torch
    import torch.nn.functional as F
    import torch_geometric
    from utils import log, pad_tensor, GraphDataset
    sys.path.insert(0, os.path.abspath(f'models'))
    from models.model_gasse import GNNPolicy

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    ### LOG ###
    logfile = os.path.join(running_dir, f'{args.model_name}_{args.train_mode}.txt')
    pklfile = os.path.join(running_dir, f'{args.model_name}_{args.train_mode}.pkl')
    if os.path.exists(logfile) or os.path.exists(pklfile):
        print(f"The log file and .pkl files already exist in the {running_dir} directory. Please delete them manually or back them up before training the model...")
        exit(1)

    log(f"Your GCNN Model: {args.train_mode}", logfile)
    log(f"max_epochs: {max_epochs}", logfile)
    log(f"batch_size: {batch_size}", logfile)
    log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    log(f"valid_batch_size : {valid_batch_size }", logfile)
    log(f"lr: {lr}", logfile)
    log(f"entropy bonus: {entropy_bonus}", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {args.seed}", logfile)


    policy = GNNPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.2)
    best_valid_loss = 1e6
    num_bad_epoch = 0

    train_files = [str(file) for file in (pathlib.Path(f'{args.root_dir}/KIDA_data/samples') / problem_folder / 'train_MC_UE').glob('sample_*.pkl')]
    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]
    valid_files = [str(file) for file in (pathlib.Path(f'{args.root_dir}/KIDA_data/samples') / problem_folder / 'valid').glob('sample_*.pkl')]
    pretrain_data = GraphDataset(pretrain_files)
    pretrain_loader = torch_geometric.loader.DataLoader(pretrain_data, pretrain_batch_size, shuffle=False, num_workers=4)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.loader.DataLoader(valid_data, valid_batch_size, shuffle=False, num_workers=4)

    log(f"pretrain samples: {len(pretrain_files)}", logfile)
    log(f"train samples:    {len(train_files)}", logfile)
    log(f"valid samples:    {len(valid_files)}", logfile)
    log(f"train path:       {args.root_dir}/KIDA_data/samples/{problem_folder}/train_MC_UE", logfile)
    log(f"valid path:       {args.root_dir}/KIDA_data/samples/{problem_folder}/valid", logfile)

    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)
        if epoch == 0:
            n = pretrain(policy, pretrain_loader)
            log(f"PRETRAINED {n} LAYERS", logfile)
        else:
            epoch_train_files = rng.choice(train_files, int(np.floor(len(train_files) / batch_size)) * batch_size, replace=True)
            train_data = GraphDataset(epoch_train_files)
            train_loader = torch_geometric.loader.DataLoader(train_data, batch_size, shuffle=True, num_workers=4)
            train_loss, train_kacc, entropy = process(policy, train_loader, top_k, args.train_mode, optimizer)
            log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)
        
        # TEST
        valid_loss, valid_kacc, entropy = process(policy, valid_loader, top_k, args.train_mode, None)
        log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

        scheduler.step(valid_loss)
        num_bad_epoch += 1
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            num_bad_epoch = 0
            torch.save(policy.state_dict(), pathlib.Path(running_dir)/f'{args.model_name}_{args.train_mode}.pkl')
            log(f"  best model so far", logfile)
        elif num_bad_epoch == 10:
            log(f"  10 epochs without improvement, decreasing learning rate, current lr: {scheduler.get_last_lr()[0]:.2e}", logfile)
        elif num_bad_epoch == 20:
            log(f"  20 epochs without improvement, early stopping, current lr: {scheduler.get_last_lr()[0]:.2e}", logfile)
            break

    policy.load_state_dict(torch.load(pathlib.Path(running_dir)/f'{args.model_name}_{args.train_mode}.pkl', weights_only=True))
    valid_loss, valid_kacc, entropy = process(policy, valid_loader, top_k, args.train_mode,None)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)
