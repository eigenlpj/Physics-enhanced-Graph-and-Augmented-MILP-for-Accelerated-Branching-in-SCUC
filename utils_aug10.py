"""
Adapted from CAMbranch (https://github.com/linjc16)
Modified for PGNN (Physics-informed graph network) branching framework under the same MIT License.
"""
import gzip
import pickle
import datetime
import random

import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric
import argparse
from torch_geometric.data import Batch
import copy
import pdb
import time


def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)


def pad_tensor(input_, pad_sizes, pad_value=-1e4):
    # input_f32 = input_.float()
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    # output = output.type_as(input_)
    return output


class BipartiteNodeData(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, nb_candidates, candidate_choice, candidate_scores):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = nb_candidates
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class BipartiteNodeData_huake_loss(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, nb_candidates, candidate_choice, candidate_scores, z_i_with_w, depth_i):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = nb_candidates
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores
        self.z_i_with_w = z_i_with_w
        self.depth_i = depth_i

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class AugmentedGraphDataset_LJM_v0(torch.utils.data.Dataset):
    def __init__(self, sample_files, num_augs=1, multiplier_dis=4, multiplier_con=20, rel_con=0.1, shift_mode='abs_shift', if_pgraph='YES'):
        super().__init__()
        self.sample_files = sample_files
        self.num_augs = num_augs  # if 0: return original only; if >=1: return (orig, aug) pairs
        self.multiplier_dis = multiplier_dis
        self.multiplier_con = multiplier_con
        self.rel_con = rel_con
        self.shift_mode = shift_mode
        self.if_pgraph = if_pgraph

    def __len__(self):
        if self.num_augs == 0:
            # Validation mode: one sample per file
            return len(self.sample_files)
        else:
            # Training mode: num_augs pairs per original sample
            return len(self.sample_files) * self.num_augs

    def augment(self, constraint_features, edge_indices, edge_features, variable_features,
                candidates, candidate_choice, candidate_scores, row_norms, has_lhs, has_rhs,
                center=0, multiplier_dis=4, multiplier_con=20, rel_con=0.1, shift_mode='abs_shift'):
        """
        Generate `self.num_augs` augmented graphs via random shifting.
        Returns a list of `num_augs` BipartiteNodeData objects (augmented only).
        Only called when num_augs >= 1.
        """
        graph_aug_list = []
        bin_v_index = (variable_features[:, 0] == 1)
        int_v_index = (variable_features[:, 1] == 1)
        imp_int_v_index = (variable_features[:, 2] == 1)
        con_v_index = (variable_features[:, 3] == 1)

        col_type_sums = variable_features[:, :4].sum(dim=1)
        assert (col_type_sums == 1).all(), ("The current sample is abnormal. "
                                            "There is an error in the variable type determination for the first four columns of a certain row. "
                                            "The variable type can only be bin, int, imp_int, or con!")

        for i in range(self.num_augs):
            shifted_vector = torch.zeros(variable_features.shape[0])
            discrete_mask = bin_v_index | int_v_index | imp_int_v_index
            shifted_vector[discrete_mask] = torch.rand(discrete_mask.sum()) * multiplier_dis + center
            if shift_mode == 'abs_shift':
                shifted_vector[con_v_index] = torch.rand(con_v_index.sum()) * multiplier_con + center
            elif shift_mode == 'rel_shift':
                solval_values = variable_features[:, 16].clone().float()
                con_base = solval_values[con_v_index]                     # shape: [num_con]
                con_shift = torch.zeros_like(con_base)                    #
                zero_mask = (con_base == 0)                               # bool, shape [num_con]
                nonzero_mask = ~zero_mask                                 # bool, shape [num_con]
                if nonzero_mask.any():
                    perturb_ratio = torch.rand(nonzero_mask.sum()) * (2 * rel_con) - rel_con
                    con_shift[nonzero_mask] = con_base[nonzero_mask] * perturb_ratio + center
                shifted_vector[con_v_index] = con_shift

            mask_size = int(variable_features.shape[0] * np.random.uniform(low=0.1, high=0.8))
            shifted_vector[torch.randperm(variable_features.shape[0])[:mask_size]] = 0

            shifted_vector[bin_v_index] = torch.round(shifted_vector[bin_v_index])
            shifted_vector[int_v_index] = torch.round(shifted_vector[int_v_index])
            shifted_vector[imp_int_v_index] = torch.round(shifted_vector[imp_int_v_index])

            var_feats_new = copy.deepcopy(variable_features)
            bin_and_nonzero = bin_v_index & (shifted_vector != 0)
            var_feats_new[bin_and_nonzero, 0] = 0
            var_feats_new[bin_and_nonzero, 1] = 1
            var_feats_new[:, 16] = var_feats_new[:, 16] + shifted_vector
            var_feats_new[:, 17] = var_feats_new[:, 17] + shifted_vector
            var_feats_new[:, 18] = var_feats_new[:, 18] + shifted_vector

            cons_feats_new = copy.deepcopy(constraint_features)
            norm_part1 = row_norms[has_lhs]
            norm_part2 = row_norms[has_rhs]
            concat_norm = np.concatenate((norm_part1, norm_part2))
            cons_feats_new[:, 1] = cons_feats_new[:, 1] * concat_norm

            indices = edge_indices
            values = edge_features.squeeze(1)
            size = (cons_feats_new.size(0), var_feats_new.size(0))
            coef_matrix_sparse = torch.sparse_coo_tensor(indices=indices, values=values, size=size)
            delta = torch.sparse.mm(coef_matrix_sparse, shifted_vector.unsqueeze(1)).squeeze(1)
            cons_feats_new[:, 1] = cons_feats_new[:, 1] + delta
            cons_feats_new[:, 1] = cons_feats_new[:, 1] / concat_norm

            graph_new = BipartiteNodeData(
                cons_feats_new, edge_indices, edge_features, var_feats_new,
                candidates, len(candidates), candidate_choice, candidate_scores
            )
            graph_new.nb_constraints = cons_feats_new.shape[0]
            graph_new.nb_variables = var_feats_new.shape[0]
            graph_new.num_nodes = cons_feats_new.shape[0] + var_feats_new.shape[0]

            graph_aug_list.append(graph_new)
        return graph_aug_list


    def __getitem__(self, idx):
        if self.num_augs == 0:

            file_path = self.sample_files[idx]
            with gzip.open(file_path, 'rb') as f:
                sample = pickle.load(f)

            sample_observation, _, sample_action, sample_action_set, sample_scores = sample['data']
            c_dict, e_dict, v_dict, norm_data = sample_observation

            constraint_features = torch.FloatTensor(c_dict['values'])
            edge_indices = torch.LongTensor(e_dict['indices'].astype(np.int32))
            edge_features = torch.FloatTensor(e_dict['values'])

            if self.if_pgraph == 'YES':
                cols_part1 = v_dict['values'][:, :19]
                cols_20 = v_dict['values'][:, 19:20]
                cols_27 = v_dict['values'][:, 26:27]
                variable_new = np.hstack([cols_part1, cols_20, cols_27])
            else:
                variable_new = v_dict['values'][:, :19]

            variable_features = torch.FloatTensor(variable_new)

            row_norms = norm_data['row_norms']
            has_lhs = norm_data['has_lhs']
            has_rhs = norm_data['has_rhs']

            candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
            candidate_choice = torch.where(candidates == sample_action)[0][0]
            candidate_scores = torch.FloatTensor(sample_scores)

            z_i = np.zeros((len(sample_action_set)))
            # delta_param = 0.1
            # tao_param = 0.5   # The label smoothing degree is 0.5
            delta_param = 0.1
            tao_param = 0       # Remove label smoothing
            max_score = np.max(sample_scores)
            border_score = (1 - delta_param) * max_score
            max_i = np.argmax(sample_scores)
            less_max_i = np.where((sample_scores >= border_score) & (sample_scores <= max_score))[0]
            equal_max_i = np.where(sample_scores == max_score)[0]
            n = len(less_max_i)
            n_max = len(equal_max_i)
            how_deal_equal_max_i = "MAX"
            how_deal_equal_max_i = "LESS"
            if n > 1:
                if how_deal_equal_max_i == "LESS":
                    z_i[less_max_i] = tao_param / (n - 1)
                    z_i[max_i] = 1.0 - tao_param
                elif how_deal_equal_max_i == "MAX":
                    if n > n_max:
                        z_i[less_max_i] = tao_param / (n - n_max)
                        z_i[equal_max_i] = (1.0 - tao_param) / n_max
                    else:
                        z_i[equal_max_i] = (1.0 - tao_param) / n_max
                        assert n == n_max, f"The current value of n must be equal to n_max; otherwise, the result of label softening will be abnormal..."
            else:
                z_i[max_i] = 1.0
            z_i_tensor = torch.FloatTensor(z_i)
            depth_float = sample['node_depth']
            depth_tensor = torch.tensor(depth_float)
            # print(f'depth_tensor:{depth_tensor}')
            orig_graph = BipartiteNodeData_huake_loss(constraint_features, edge_indices, edge_features,
                                                      variable_features,
                                                      candidates, len(candidates), candidate_choice, candidate_scores,
                                                      z_i_tensor, depth_tensor)
            # orig_graph = BipartiteNodeData(
            #     constraint_features, edge_indices, edge_features, variable_features,
            #     candidates, len(candidates), candidate_choice, candidate_scores
            # )
            orig_graph.nb_constraints = constraint_features.shape[0]
            orig_graph.nb_variables = variable_features.shape[0]
            orig_graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]

            return orig_graph  # ← single Data object, not list

        else:

            base_idx = idx // self.num_augs
            aug_idx = idx % self.num_augs

            file_path = self.sample_files[base_idx]
            with gzip.open(file_path, 'rb') as f:
                sample = pickle.load(f)

            sample_observation, _, sample_action, sample_action_set, sample_scores = sample['data']
            c_dict, e_dict, v_dict, norm_data = sample_observation

            constraint_features = torch.FloatTensor(c_dict['values'])
            edge_indices = torch.LongTensor(e_dict['indices'].astype(np.int32))
            edge_features = torch.FloatTensor(e_dict['values'])

            if self.if_pgraph == 'YES':
                cols_part1 = v_dict['values'][:, :19]
                cols_20 = v_dict['values'][:, 19:20]
                cols_27 = v_dict['values'][:, 26:27]
                variable_new = np.hstack([cols_part1, cols_20, cols_27])

            else:
                variable_new = v_dict['values'][:, :19]

            variable_features = torch.FloatTensor(variable_new)

            row_norms = norm_data['row_norms']
            has_lhs = norm_data['has_lhs']
            has_rhs = norm_data['has_rhs']

            candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
            candidate_choice = torch.where(candidates == sample_action)[0][0]
            candidate_scores = torch.FloatTensor(sample_scores)

            z_i = np.zeros((len(sample_action_set)))
            # delta_param = 0.1
            # tao_param = 0.5  # The label smoothing degree is 0.5
            delta_param = 0.1
            tao_param = 0      # Remove label smoothing
            max_score = np.max(sample_scores)
            border_score = (1 - delta_param) * max_score
            max_i = np.argmax(sample_scores)
            less_max_i = np.where((sample_scores >= border_score) & (sample_scores <= max_score))[0]
            equal_max_i = np.where(sample_scores == max_score)[0]
            n = len(less_max_i)
            n_max = len(equal_max_i)
            how_deal_equal_max_i = "MAX"
            how_deal_equal_max_i = "LESS"
            if n > 1:
                if how_deal_equal_max_i == "LESS":
                    z_i[less_max_i] = tao_param / (n - 1)
                    z_i[max_i] = 1.0 - tao_param
                elif how_deal_equal_max_i == "MAX":
                    if n > n_max:
                        z_i[less_max_i] = tao_param / (n - n_max)
                        z_i[equal_max_i] = (1.0 - tao_param) / n_max
                    else:
                        z_i[equal_max_i] = (1.0 - tao_param) / n_max
                        assert n == n_max, f"The current value of n must be equal to n_max; otherwise, the result of label softening will be abnormal..."
            else:
                z_i[max_i] = 1.0
            z_i_tensor = torch.FloatTensor(z_i)
            depth_float = sample['node_depth']
            depth_tensor = torch.tensor(depth_float)
            # print(f'depth_tensor:{depth_tensor}')
            orig_graph = BipartiteNodeData_huake_loss(constraint_features, edge_indices, edge_features, variable_features,
                                                 candidates, len(candidates), candidate_choice, candidate_scores,
                                                 z_i_tensor, depth_tensor)

            orig_graph.nb_constraints = constraint_features.shape[0]
            orig_graph.nb_variables = variable_features.shape[0]
            orig_graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]

            aug_graphs = self.augment(
                constraint_features, edge_indices, edge_features, variable_features,
                candidates, candidate_choice, candidate_scores,
                row_norms, has_lhs, has_rhs,
                center=0, multiplier_dis=self.multiplier_dis, multiplier_con=self.multiplier_con, rel_con=self.rel_con,shift_mode=self.shift_mode
            )

            selected_aug = aug_graphs[aug_idx]

            return [orig_graph, selected_aug]  # ← list of two graphs


def collate_fn_aug(batch):

    if isinstance(batch[0], list):
        batch_data = []
        batch_augmented_data = []
        for sample in batch:
            batch_data.append(sample[0])
            batch_augmented_data.append(sample[1])
        data_batch = Batch.from_data_list(batch_data)
        augmented_data_batch = Batch.from_data_list(batch_augmented_data)
        return data_batch, augmented_data_batch
    else:
        batch_data = torch_geometric.data.Batch.from_data_list(batch)
        return batch_data


def valid_seed(seed):
    """Check whether seed is a valid random seed or not."""
    seed = int(seed)
    if seed < 0 or seed > 2**32 - 1:
        raise argparse.ArgumentTypeError(
                "seed must be any integer between 0 and 2**32 - 1 inclusive")
    return seed