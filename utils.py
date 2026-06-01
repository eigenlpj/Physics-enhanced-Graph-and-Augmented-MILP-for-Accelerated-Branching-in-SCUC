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
from sklearn.preprocessing import StandardScaler


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


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, _, sample_action, sample_action_set, sample_scores = sample['data']
        c_dict, e_dict, v_dict, _ = sample_observation
        constraint_features = c_dict['values']
        edge_indices = e_dict['indices']
        edge_features = e_dict['values']
        variable_features = v_dict['values']


        constraint_features = torch.FloatTensor(constraint_features)
        edge_indices = torch.LongTensor(edge_indices.astype(np.int32))
        edge_features = torch.FloatTensor(edge_features)
        variable_features = torch.FloatTensor(variable_features[:, :19])
        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_choice = torch.where(candidates == sample_action)[0][0]  # action index relative to candidates
        candidate_scores = torch.FloatTensor(sample_scores)

        z_i = np.zeros((len(sample_action_set)))
        # delta_param = 0.5
        # tao_param = 0.5
        delta_param = 0.1
        tao_param = 0.5    #
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
        graph = BipartiteNodeData_huake_loss(constraint_features, edge_indices, edge_features, variable_features,
                                  candidates, len(candidates), candidate_choice, candidate_scores, z_i_tensor, depth_tensor)

        # graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
        #                           candidates, len(candidates), candidate_choice, candidate_scores)
        
        graph.nb_constraints = constraint_features.shape[0]
        graph.nb_variables = variable_features.shape[0]
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        
        return graph


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