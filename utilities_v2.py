"""
Adapted from ds4dm/learn2branch (https://github.com/ds4dm/learn2branch/blob/master/02_generate_dataset.py) and CAMbranch (https://github.com/linjc16)
Modified for PGNN (Physics-informed graph network) branching framework under the same MIT License.
"""
import datetime
import numpy as np
import scipy.sparse as sp
import pyscipopt as scip
import pickle
import gzip
import argparse


def log(msg, logfile=None):
    msg = f'[{datetime.datetime.now()}] {msg}'
    print(msg)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(msg, file=f)


def init_scip_params_collect(model, seed, heuristics=True, presolving=True, separating=True, conflict=True):

    seed = seed % 2147483648  # SCIP seed range

    # set up randomization
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)

    model.setIntParam('display/verblevel', 0)
    # model.setIntParam('display/freq', 1)

    # separation only at root node
    model.setIntParam('separating/maxrounds', 0)

    # no restart
    model.setIntParam('presolving/maxrestarts', 0)  #

    # if asked, disable presolving
    if not presolving:
        model.setIntParam('presolving/maxrounds', 0)
        model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable separating (cuts)
    if not separating:
        model.setIntParam('separating/maxroundsroot', 0)

    # if asked, disable conflict analysis (more cuts)
    if not conflict:
        model.setBoolParam('conflict/enable', False)

    # if asked, disable primal heuristics
    if not heuristics:
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)


def init_scip_params_evaluate(model, seed, heuristics=True, presolving=True, separating=True, conflict=True):

    seed = seed % 2147483648  # SCIP seed range

    # set up randomization
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)

    model.setIntParam('display/verblevel', 0)
    # model.setIntParam('display/freq', 3)

    # separation only at root node
    model.setIntParam('separating/maxrounds', 0)

    # no restart
    model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable presolving
    if not presolving:
        model.setIntParam('presolving/maxrounds', 0)
        model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable separating (cuts)
    if not separating:
        model.setIntParam('separating/maxroundsroot', 0)

    # if asked, disable conflict analysis (more cuts)
    if not conflict:
        model.setBoolParam('conflict/enable', False)

    # if asked, disable primal heuristics
    if not heuristics:
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)


def init_scip_params_relpscost(model, seed, heuristics=True, presolving=True, separating=True, conflict=True):

    seed = seed % 2147483648  # SCIP seed range

    # set up randomization
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)

    model.setIntParam('display/verblevel', 0)
    # model.setIntParam('display/freq', 1)

    # separation only at root node
    model.setIntParam('separating/maxrounds', 0)

    # no restart
    model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable presolving
    if not presolving:
        model.setIntParam('presolving/maxrounds', 0)
        model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable separating (cuts)
    if not separating:
        model.setIntParam('separating/maxroundsroot', 0)

    # if asked, disable conflict analysis (more cuts)
    if not conflict:
        model.setBoolParam('conflict/enable', False)

    # if asked, disable primal heuristics
    if not heuristics:
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)


def extract_state(model, buffer=None):
    """
    Compute a bipartite graph representation of the solver. In this
    representation, the variables and constraints of the MILP are the
    left- and right-hand side nodes, and an edge links two nodes iff the
    variable is involved in the constraint. Both the nodes and edges carry
    features.

    Parameters
    ----------
    model : pyscipopt.scip.Model
        The current model.
    buffer : dict
        A buffer to avoid re-extracting redundant information from the solver
        each time.
    Returns
    -------
    variable_features : dictionary of type {'names': list, 'values': np.ndarray}
        The features associated with the variable nodes in the bipartite graph.
    edge_features : dictionary of type ('names': list, 'indices': np.ndarray, 'values': np.ndarray}
        The features associated with the edges in the bipartite graph.
        This is given as a sparse matrix in COO format.
    constraint_features : dictionary of type {'names': list, 'values': np.ndarray}
        The features associated with the constraint nodes in the bipartite graph.
    norm_data: Inverse normalization for some feature values
    """
    if buffer is None or model.getNNodes() == 1:
        buffer = {}

    # update state from buffer if any
    s = model.getState(buffer['scip_state'] if 'scip_state' in buffer else None)
    buffer['scip_state'] = s

    if 'state' in buffer:
        obj_norm = buffer['state']['obj_norm']
    else:
        obj_norm = np.linalg.norm(s['col']['coefs'])
        obj_norm = 1 if obj_norm <= 0 else obj_norm

    row_norms = s['row']['norms']
    row_norms[row_norms == 0] = 1

    # Column features
    n_cols = len(s['col']['types'])

    if 'state' in buffer:
        col_feats = buffer['state']['col_feats']
    else:
        col_feats = {}
        col_feats['type'] = np.zeros((n_cols, 4))  # BINARY INTEGER IMPLINT CONTINUOUS
        col_feats['type'][np.arange(n_cols), s['col']['types']] = 1
        col_feats['coef_normalized'] = s['col']['coefs'].reshape(-1, 1) / obj_norm

    col_feats['has_lb'] = ~np.isnan(s['col']['lbs']).reshape(-1, 1)
    col_feats['has_ub'] = ~np.isnan(s['col']['ubs']).reshape(-1, 1)
    col_feats['sol_is_at_lb'] = s['col']['sol_is_at_lb'].reshape(-1, 1)
    col_feats['sol_is_at_ub'] = s['col']['sol_is_at_ub'].reshape(-1, 1)
    col_feats['sol_frac'] = s['col']['solfracs'].reshape(-1, 1)
    col_feats['sol_frac'][s['col']['types'] == 3] = 0  # continuous have no fractionality
    col_feats['basis_status'] = np.zeros((n_cols, 4))  # LOWER BASIC UPPER ZERO
    col_feats['basis_status'][np.arange(n_cols), s['col']['basestats']] = 1
    col_feats['reduced_cost'] = s['col']['redcosts'].reshape(-1, 1) / obj_norm
    col_feats['age'] = s['col']['ages'].reshape(-1, 1) / (s['stats']['nlps'] + 5)
    col_feats['sol_val'] = s['col']['solvals'].reshape(-1, 1)
    col_feats['inc_val'] = s['col']['incvals'].reshape(-1, 1)
    col_feats['avg_inc_val'] = s['col']['avgincvals'].reshape(-1, 1)

    col_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in col_feats.items()]
    col_feat_names = [n for names in col_feat_names for n in names]
    col_feat_vals = np.concatenate(list(col_feats.values()), axis=-1)

    variable_features = {
        'names': col_feat_names,
        'values': col_feat_vals,}

    # Row features

    if 'state' in buffer:
        row_feats = buffer['state']['row_feats']
        has_lhs = buffer['state']['has_lhs']
        has_rhs = buffer['state']['has_rhs']
    else:
        row_feats = {}
        has_lhs = np.nonzero(~np.isnan(s['row']['lhss']))[0]
        has_rhs = np.nonzero(~np.isnan(s['row']['rhss']))[0]
        row_feats['obj_cosine_similarity'] = np.concatenate((
            -s['row']['objcossims'][has_lhs],
            +s['row']['objcossims'][has_rhs])).reshape(-1, 1)
        row_feats['bias'] = np.concatenate((
            -(s['row']['lhss'] / row_norms)[has_lhs],
            +(s['row']['rhss'] / row_norms)[has_rhs])).reshape(-1, 1)

    row_feats['is_tight'] = np.concatenate((
        s['row']['is_at_lhs'][has_lhs],
        s['row']['is_at_rhs'][has_rhs])).reshape(-1, 1)

    row_feats['age'] = np.concatenate((
        s['row']['ages'][has_lhs],
        s['row']['ages'][has_rhs])).reshape(-1, 1) / (s['stats']['nlps'] + 5)

    # # redundant with is_tight
    # tmp = s['row']['basestats']  # LOWER BASIC UPPER ZERO
    # tmp[s['row']['lhss'] == s['row']['rhss']] = 4  # LOWER == UPPER for equality constraints
    # tmp_l = tmp[has_lhs]
    # tmp_l[tmp_l == 2] = 1  # LHS UPPER -> BASIC
    # tmp_l[tmp_l == 4] = 2  # EQU UPPER -> UPPER
    # tmp_l[tmp_l == 0] = 2  # LHS LOWER -> UPPER
    # tmp_r = tmp[has_rhs]
    # tmp_r[tmp_r == 0] = 1  # RHS LOWER -> BASIC
    # tmp_r[tmp_r == 4] = 2  # EQU LOWER -> UPPER
    # tmp = np.concatenate((tmp_l, tmp_r)) - 1  # BASIC UPPER ZERO
    # row_feats['basis_status'] = np.zeros((len(has_lhs) + len(has_rhs), 3))
    # row_feats['basis_status'][np.arange(len(has_lhs) + len(has_rhs)), tmp] = 1

    tmp = s['row']['dualsols'] / (row_norms * obj_norm)
    row_feats['dualsol_val_normalized'] = np.concatenate((
            -tmp[has_lhs],
            +tmp[has_rhs])).reshape(-1, 1)

    row_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in row_feats.items()]
    row_feat_names = [n for names in row_feat_names for n in names]
    row_feat_vals = np.concatenate(list(row_feats.values()), axis=-1)

    constraint_features = {
        'names': row_feat_names,
        'values': row_feat_vals,}

    # Edge features
    if 'state' in buffer:
        edge_row_idxs = buffer['state']['edge_row_idxs']
        edge_col_idxs = buffer['state']['edge_col_idxs']
        edge_feats = buffer['state']['edge_feats']
    else:
        coef_matrix = sp.csr_matrix(
            (s['nzrcoef']['vals'] / row_norms[s['nzrcoef']['rowidxs']],
            (s['nzrcoef']['rowidxs'], s['nzrcoef']['colidxs'])),
            shape=(len(s['row']['nnzrs']), len(s['col']['types'])))
        coef_matrix = sp.vstack((
            -coef_matrix[has_lhs, :],
            coef_matrix[has_rhs, :])).tocoo(copy=False)

        edge_row_idxs, edge_col_idxs = coef_matrix.row, coef_matrix.col
        edge_feats = {}

        edge_feats['coef_normalized'] = coef_matrix.data.reshape(-1, 1)

    edge_feat_names = [[k, ] if v.shape[1] == 1 else [f'{k}_{i}' for i in range(v.shape[1])] for k, v in edge_feats.items()]
    edge_feat_names = [n for names in edge_feat_names for n in names]
    edge_feat_indices = np.vstack([edge_row_idxs, edge_col_idxs])
    edge_feat_vals = np.concatenate(list(edge_feats.values()), axis=-1)

    edge_features = {
        'names': edge_feat_names,
        'indices': edge_feat_indices,
        'values': edge_feat_vals,}

    if 'state' not in buffer:
        buffer['state'] = {
            'obj_norm': obj_norm,
            'col_feats': col_feats,
            'row_feats': row_feats,
            'has_lhs': has_lhs,
            'has_rhs': has_rhs,
            'edge_row_idxs': edge_row_idxs,
            'edge_col_idxs': edge_col_idxs,
            'edge_feats': edge_feats,
        }

    norm_data = {
        'obj_norm': obj_norm,
        'has_lhs': has_lhs,
        'has_rhs': has_rhs,
        'row_norms': row_norms,
    }

    return constraint_features, edge_features, variable_features, norm_data


def valid_seed(seed):
    """Check whether seed is a valid random seed or not."""
    seed = int(seed)
    if seed < 0 or seed > 2**32 - 1:
        raise argparse.ArgumentTypeError(
                "seed must be any integer between 0 and 2**32 - 1 inclusive")
    return seed


