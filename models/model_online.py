import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import numpy as np
from torch_geometric.utils import scatter, softmax
import copy

import random
import pdb


class PreNormException(Exception):
    pass


class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super().__init__()
        assert shift or scale
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale

        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input_.shape[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."

        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size())/self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg

        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    def __init__(self):
        super().__init__('add')
        emb_size = 64
        
        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(1, shift=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size)
        )

        self.post_conv_module = torch.nn.Sequential(
            PreNormLayer(1, shift=False)
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2*emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]), 
                                node_features=(left_features, right_features), edge_features=edge_features)
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(self.feature_module_left(node_features_i) 
                                           + self.feature_module_edge(edge_features) 
                                           + self.feature_module_right(node_features_j))
        return output


class BaseModel(torch.nn.Module):
    """
    Our base model class, which implements pre-training methods.
    """

    def pre_train_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_train_next(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def pre_train(self, *args, **kwargs):
        try:
            with torch.no_grad():
                self.forward(*args, **kwargs)
            return False
        except PreNormException:
            return True


class GNNPolicy(BaseModel):
    def __init__(self, cl=False):
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19    # 源码内容
        # var_nfeats = 21      # cambranch的v+Pmax的比例+Pmin的数值   (最好)
        # var_nfeats = 23    # cambranch的v+Pmax信息+Pmin信息
        # var_nfeats = 22    # cambranch的v+Pmax信息+keepT信息
        # var_nfeats = 25    # cambranch的v+ramp+su+sd
        # var_nfeats = 27    # cambranch的v+ptdf+plinemax(4列ptdf,4列plinemax)
        # var_nfeats = 38    # cambranch的v+Pinfo-all
        # var_nfeats = 22    # cambranch的v+Pmax的比例+Pmax的数值+Pmin的数值

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )
        
        if cl:  
            self.var_aggr_mlp = nn.Linear(emb_size * 2, emb_size)
            self.cons_aggr_mlp = nn.Linear(emb_size * 2, emb_size)
            self.last = nn.Sequential(
                nn.Linear(emb_size * 2, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size),
            )
    
    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
        
        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        output = self.output_module(variable_features).squeeze(-1)
        return output

    def forward_with_graph_embs(self, constraint_features, edge_indices, edge_features, variable_features, nb_constraints, nb_variables):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)

        variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)
        
        var_index = torch.arange(nb_variables.size(0), device=nb_variables.device).repeat_interleave(nb_variables)
        var_feats_max = scatter(variable_features, var_index, dim=0, reduce='max')
        var_feats_mean = scatter(variable_features, var_index, dim=0, reduce='mean')
        var_feats_aggr = self.var_aggr_mlp(torch.cat([var_feats_max, var_feats_mean], dim=-1))
        
        cons_index = torch.arange(nb_constraints.size(0), device=nb_constraints.device).repeat_interleave(nb_constraints)
        cons_feats_max = scatter(constraint_features, cons_index, dim=0, reduce='max')
        cons_feats_mean = scatter(constraint_features, cons_index, dim=0, reduce='mean')    
        cons_feats_aggr = self.cons_aggr_mlp(torch.cat([cons_feats_max, cons_feats_mean], dim=-1))
        
        graph_emb = self.last(torch.cat([var_feats_aggr, cons_feats_aggr], dim=-1))
        
        output = self.output_module(variable_features).squeeze(-1)
        
        return output, graph_emb
