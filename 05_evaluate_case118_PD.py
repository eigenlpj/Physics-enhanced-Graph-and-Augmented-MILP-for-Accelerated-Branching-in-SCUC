"""
Adapted from ds4dm/learn2branch (https://github.com/ds4dm/learn2branch/blob/master/02_generate_dataset.py) and CAMbranch (https://github.com/linjc16)
Modified for PGNN (Physics-informed graph network) branching framework under the same MIT License.
"""
import os
import importlib
import argparse
import csv
import numpy as np
import time
import pickle
import gzip
import subprocess
import pdb
import torch
from models.model_online import GNNPolicy             
from models.model_online_Pinfo import GNNPolicy_Pinfo 
import utilities_v2  
import pyscipopt as scip
import utilities_pinfo  
import IEEE_g     

# MODEL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PolicyBranching(scip.Branchrule):

    def __init__(self, policy, unitlist, load, linelist, Blocklineindex, PTDF_dict, online_sample_dir, do_pscost, case_name, train_freq):
        super().__init__()

        self.policy_type = policy['type']
        self.policy_name = policy['name']


        # # ***Should the intermediate information of the solution process be output?
        # self.time_record_list = []
        # self.primal_record_list = []
        # self.dual_record_list = []
        # self.relgap_record_list = []
        # self.nnodes_record_list = []
        # self.bvarname_list = []
        # self.genid_record_list = []
        # self.t_record_list = []
        # self.dual_integral_list = []
        # self.dual_integral = 0.0


        self.unitlist = unitlist
        self.load = load
        self.linelist = linelist
        self.Blocklineindex = Blocklineindex
        self.PTDF_dict = PTDF_dict
        self.branch_begin_time = 0.0

        self.sample_counter = 0
        self.current_dual = -float('inf')
        self.delta_dual = 0.0
        self.total_delta_dual = 0.0
        self.do_pscost = do_pscost
        self.online_sample_dir = online_sample_dir
        self.train_threshold_num = 100
        self.train_threshold_ratio = 0.1
        self.train_span = 1000
        # self.gcnn_i_model = GNNPolicy(cl=True).to(MODEL_DEVICE)
        self.gcnn_i_model = GNNPolicy(cl=True).to(device)
        self.case_name = case_name
        self.old_model_file = 'Blank Name'
        self.train_freq = train_freq
        self.branch_count = 0

        if self.policy_type == 'gnn':
            self.policy = policy['model']

            # MODEL_DEVICE = next(model.parameters()).device
            # print(f"model is on device: {MODEL_DEVICE}")  # DEBUG ljm

        elif self.policy_type == 'internal':
            self.policy = policy['name']

        else:
            raise NotImplementedError

    def branchinitsol(self):
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.state_buffer = {}
        self.khalil_root_buffer = {}
        self.pinfo_buffer = None
        self.fea_time = 0
        self.inf_time = 0
        self.old_NLPRows = 0
        self.old_NLPCols = 0

    def branchexeclp(self, allowaddcons):

        if self.branch_count == 0:
            self.branch_begin_time = self.model.getTotalTime()

        # SCIP internal branching rule
        if self.policy_type == 'internal':
            result = self.model.executeBranchRule(self.policy, allowaddcons)
            if policy['name'] == 'vanillafullstrong':
                assert result == scip.SCIP_RESULT.DIDNOTRUN
                cands, scores, npriocands, bestcand = self.model.getVanillafullstrongData()
                best_var = cands[bestcand]                                       # type, pyscipopt.scip.variable
                best_var_Id = utilities_pinfo.parse_candidate_vars([best_var.name])
                self.model.branchVar(best_var)
                result = scip.SCIP_RESULT.BRANCHED


            # # ***Should the intermediate information of the solution process be output?
            # self.time_record_list.append(self.model.getTotalTime())
            # self.primal_record_list.append(self.model.getPrimalbound())
            # self.dual_record_list.append(self.model.getDualbound())
            # if len(self.dual_record_list) > 1:
            #     self.dual_integral += (self.time_record_list[-1] - self.time_record_list[-2]) * (self.dual_record_list[-1]-self.dual_record_list[0])
            #     self.dual_integral_list.append(self.dual_integral)
            # else:
            #     self.dual_integral_list.append(0.0)
            # self.relgap_record_list.append(self.model.getGap())
            # self.nnodes_record_list.append(self.model.getNNodes())
            # if policy['name'] == 'vanillafullstrong':
            #     self.bvarname_list.append(best_var_Id[0].name)
            #     self.genid_record_list.append(best_var_Id[0].genid)
            #     self.t_record_list.append(best_var_Id[0].t)
            # else:
            #     self.bvarname_list.append(0)
            #     self.genid_record_list.append(0)
            #     self.t_record_list.append(0)


        # custom policy branching
        else:

            start_time = time.perf_counter()

            current_rows = self.model.getNLPRows()
            current_cols = self.model.getNLPCols()
            if current_rows != self.old_NLPRows or current_cols != self.old_NLPCols:  # When the number of rows OR columns changes, rebuild the state from scratch.
                # print(f"c_rows:{current_rows}, old_rows:{self.old_NLPRows}\n"
                #       f"c_cols:{current_cols}, old_cols:{self.old_NLPCols}")        # ljm debug
                dummy_buffer = {}
                state = utilities_v2.extract_state(self.model, dummy_buffer)
                self.old_NLPRows = current_rows  # Sync the LP dimensions to detect structural changes later.
                self.old_NLPCols = current_cols
                self.state_buffer = dummy_buffer  # Sync state_buffer
            else:
                # When both rows and columns remain constant, reuse the buffer to save time.
                state = utilities_v2.extract_state(self.model, self.state_buffer)

            c_fea, e_fea, v_fea, norm_data = state
            if self.pinfo_buffer is None:
                varlist = self.model.getVars(transformed=True)
                varnamelist = [v.name for v in varlist]
                Pinfo_array = np.zeros((len(varnamelist), 2))
                assert v_fea['values'].shape[0] == len(varnamelist), 'variables from gasse ≠ variables from pyscipopt!'
                varId = utilities_pinfo.parse_candidate_vars(varnamelist)
                n_units = len(self.unitlist)
                n_ts = len(self.load)
                pmax = np.array([u.pmax for u in self.unitlist])
                pmin = np.array([u.pmin for u in self.unitlist])
                INTEGER_VAR_TYPES = {'uit', 'yit', 'zit', 'ycoldit'}
                mask = np.array(
                    [var.name in INTEGER_VAR_TYPES for var in varId])  # (N,)
                gen_ids = np.array([var.genid - 1 for var in varId])  # (N,)
                gen_ids_valid = gen_ids[mask]  # (M,)
                t_ids = np.array([var.t - 1 for var in varId])  # (N,)
                t_ids_valid = t_ids[mask]  # (M,)
                load_sums = np.array([self.load[t].LoadSum for t in t_ids_valid])  # (M,)
                assert (gen_ids >= -1).all() and (
                        gen_ids < n_units).all(), f"genid {gen_ids}  is out of the range of the unitlist {n_units}!"
                assert (t_ids >= -1).all() and (t_ids < n_ts).all(), f"t {t_ids}  is out of the range of load {n_ts}!"
                Pinfo_array[mask, 0] = pmax[gen_ids_valid] / load_sums
                Pinfo_array[mask, 1] = pmin[gen_ids_valid]
                self.pinfo_buffer = Pinfo_array
            v_fea_Pinfo = np.concatenate([v_fea['values'], self.pinfo_buffer], axis=1)  # (N, R+i)
            end_time = time.perf_counter()
            self.fea_time += (end_time - start_time)

            ## Whether to consider integer LP solutions as candidate branching variables is:
            ## if .getPseudoBranchCands() is called; otherwise, getLPBranchCands() is called.
            # candidate_vars, *_ = self.model.getPseudoBranchCands()
            candidate_vars, *_ = self.model.getLPBranchCands()
            candidate_mask = [var.getCol().getLPPos() for var in candidate_vars]

            if len(candidate_vars) == 1:
                best_var = candidate_vars[0]
            elif self.policy_type == 'gnn':
                start_time = time.perf_counter()
                GNNPolicy_Pinfo_list =  ['IGNN_NoAMILP', 'gcnn_aug',
                                         'IGNN_NoAMILP_5k', 'gcnn_aug_5k',
                                         'IGNN_NoAMILP_80k', 'gcnn_aug_80k',
                                         ]
                if self.policy_name in GNNPolicy_Pinfo_list:
                    variable_new = v_fea_Pinfo
                else:
                    variable_new = v_fea['values']
                state_pt = (
                    # torch.tensor(c_fea['values'], dtype=torch.float32).to(device=MODEL_DEVICE),  # float
                    # torch.tensor(e_fea['indices'], dtype=torch.int64).to(device=MODEL_DEVICE),  # long
                    # torch.tensor(e_fea['values'], dtype=torch.float32).to(device=MODEL_DEVICE),  # float
                    # torch.tensor(variable_new, dtype=torch.float32).to(device=MODEL_DEVICE),  # float
                    torch.tensor(c_fea['values'], dtype=torch.float32).to(device=device),  # float
                    torch.tensor(e_fea['indices'], dtype=torch.int64).to(device=device),  # long
                    torch.tensor(e_fea['values'], dtype=torch.float32).to(device=device),  # float
                    torch.tensor(variable_new, dtype=torch.float32).to(device=device),  # float
                )
                with torch.no_grad():
                    var_logits = self.policy(*state_pt).cpu().numpy()
                candidate_scores = var_logits[candidate_mask]
                best_var = candidate_vars[candidate_scores.argmax()]
                end_time = time.perf_counter()
                self.inf_time += (end_time - start_time)


            else:
                raise NotImplementedError

            # # ***Should the intermediate information of the solution process be output?
            # self.time_record_list.append(self.model.getTotalTime())
            # self.primal_record_list.append(self.model.getPrimalbound())
            # self.dual_record_list.append(self.model.getDualbound())
            # if len(self.dual_record_list) > 1:
            #     self.dual_integral += (self.time_record_list[-1] - self.time_record_list[-2]) * (self.dual_record_list[-1]-self.dual_record_list[0])
            #     self.dual_integral_list.append(self.dual_integral)
            # else:
            #     self.dual_integral_list.append(0.0)
            # self.relgap_record_list.append(self.model.getGap())
            # self.nnodes_record_list.append(self.model.getNNodes())
            # best_var_Id = utilities_pinfo.parse_candidate_vars([best_var.name])
            # self.bvarname_list.append(best_var_Id[0].name)
            # self.genid_record_list.append(best_var_Id[0].genid)
            # self.t_record_list.append(best_var_Id[0].t)


            self.model.branchVar(best_var)
            result = scip.SCIP_RESULT.BRANCHED

        self.branch_count += 1

        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result': result}

    def branchexecext(self, allowaddcons):
        return {"result": scip.SCIP_RESULT.DIDNOTFIND}

    def branchexecps(self, allowaddcons):
        return {"result": scip.SCIP_RESULT.DIDNOTFIND}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset', 'case118', 'case1888', '24GX', 'case2383', 'case2848'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--type',
        nargs='+',
        choices=['small', 'medium', 'big']
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default=''
    )
    parser.add_argument(
        '--internal-brancher',
        nargs='+',
        choices=['relpscost', 'vanillafullstrong', 'fullstrong']
    )
    parser.add_argument(
        '--ml-model',
        nargs='+',
        choices=['extratrees_gcnn_agg', 'lambdamart_khalil', 'svmrank_khalil']
    )
    parser.add_argument(
        '--gnn-model',
        nargs='+',
        choices=['gcnn', 'gcnn_0.1', 'gcnn_huake', 'IGNN_NoPgraph', 'IGNN_NoAMILP', 'gcnn_aug',
                 'gcnn_0.1_5k', 'gcnn_huake_5k', 'IGNN_NoAMILP_5k', 'gcnn_aug_5k',
                 'gcnn_0.1_80k', 'gcnn_huake_80k', 'IGNN_NoAMILP_80k', 'gcnn_aug_80k',
                 ]
    )
    parser.add_argument('--alpha-cl', type=float, default=0.05)
    parser.add_argument('--alpha-reg', type=float, default=0.01)
    parser.add_argument(
        '--online_sample_dir',
        type=str,
        default = f'D:/LiJiamigFile/CAMBranch-ljmdata/online_data/samples'
    )

    parser.add_argument(
        '--train_freq',
        help='The number of branches required for one training session.',
        type=int,
        default=2000,
    )

    parser.add_argument(
        '--do_pscost',
        help='Do you want the GCNN_AUG to be trained online based on pscost information? Please choose YES or NO.',
        type=str,
        choices=['YES', 'NO'],
        default='NO',
    )

    args = parser.parse_args()
    internal_branchers = [] if args.internal_brancher is None else args.internal_brancher
    other_models = [] if args.ml_model is None else args.ml_model
    gnn_models = [] if args.gnn_model is None else args.gnn_model
    os.makedirs(args.online_sample_dir)
    instances = []

    # seeds = [2026, 5, 20]
    seeds = [2026]
    time_limit = 1800.0
    gap_limit = 0.00000

    if len(args.prefix) == 0:
        args.prefix = '_'.join(internal_branchers + other_models + gnn_models)
    result_file = f"{args.problem}_{args.prefix}_{time.strftime('%Y%m%d-%H%M%S')}.csv"

    if args.problem == 'case118':

        #                    I9
        # user_selected_ids = [2532 ]

        #                    I1    I2    I3    I4    I5    I6    I7    I8    I9    I10
        user_selected_ids = [2404, 2405, 2433, 2434, 2435, 2501, 2504, 2505, 2532, 2534, 2550, 2560, 2570, 2580, 2590 ]
        instances += [{'type': f'{args.gpu}(-1:cpu,others:gpu)', 'path': f"D:/LiJiamigFile/CAMBranch-ljmdata/data/instances/case118-v2/test_milp/case118_{i}/case118_{i}.lp"} for i in user_selected_ids]

        # # additional 50 instances
        # user_selected_ids = range(2551, 2601)
        # instances += [{'type': f'{args.gpu}(-1:cpu,others:gpu)', 'path': f"D:/LiJiamigFile/CAMBranch-ljmdata/data/instances/case118-v2/test_milp/case118_{i}/case118_{i}.lp"} for i in user_selected_ids]

    else:
        raise NotImplementedError

    branching_policies = []
    
    # GNN models
    for model in gnn_models:
        for seed in seeds:
            branching_policies.append({
                'type': 'gnn',
                'name': model,
                'seed': seed,
            })
    
    # SCIP internal brancher baselines
    for brancher in internal_branchers:
        for seed in seeds:
            branching_policies.append({
                    'type': 'internal',
                    'name': brancher,
                    'seed': seed,
             })

    print(f"problem: {args.problem}")
    print(f"milp_num: {len(instances)}")
    print(f"gpu: {args.gpu}")
    print(f"time limit: {time_limit} s")
    print(f'gap limit: {gap_limit:.4f}')

    ### PYTORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = 'cpu'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:{args.gpu}"

    # load and assign tensorflow models to policies (share models and update parameters)
    loaded_models = {}
    loaded_calls = {}
    for policy in branching_policies:
        if policy['type'] == 'gnn':
            if policy['name'] not in loaded_models:
                
                if policy['name'] == 'gcnn' or policy['name'] == 'gcnn_0.1':
                    model = GNNPolicy(cl=False)

                    # case118
                    model.load_state_dict(
                        torch.load(r"D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_0.1_gasse\10K\gcnn_0.1_gasse.pkl",
                                   map_location=device,
                                   weights_only=True)
                    )
                    print(r"load model 1: D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_0.1_gasse\10K\gcnn_0.1_gasse.pkl")
                    model = model.to(device)
                    model.eval()
                elif policy['name'] == 'gcnn_huake':
                    model = GNNPolicy(cl=False)

                    # case118
                    model.load_state_dict(
                        torch.load(r"D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_0.1_huake\10K\gcnn_0.1_huake.pkl",
                                   map_location=device,
                                   weights_only=True)
                    )
                    print(r"load model 2: D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_0.1_huake\10K\gcnn_0.1_huake.pkl")
                    model = model.to(device)
                    model.eval()
                elif policy['name'] == 'IGNN_NoPgraph':
                    model = GNNPolicy(cl=True)

                    # case118
                    model.load_state_dict(
                        torch.load(r"D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_aug\NoPgraph\10K\gcnn_aug_1.pkl",
                                   map_location=device,
                                   weights_only=True)
                    )
                    print(r"load model 3: D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_aug\NoPgraph\10K\gcnn_aug_1.pkl")
                    model = model.to(device)
                    model.eval()
                elif policy['name'] == 'IGNN_NoAMILP':
                    model = GNNPolicy_Pinfo(cl=True)

                    # case118
                    model.load_state_dict(
                        torch.load(r"D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_aug\NoAMILP\10K\gcnn_aug_1.pkl",
                                   map_location=device,
                                   weights_only=True)
                    )
                    print(r"load model 4: D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_aug\NoAMILP\10K\gcnn_aug_1.pkl")
                    model = model.to(device)
                    model.eval()
                elif policy['name'] == 'gcnn_aug':
                    model = GNNPolicy_Pinfo(cl=True)

                    # case118
                    model.load_state_dict(
                        torch.load(r"D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_aug\PGNN\10K\gcnn_aug_1.pkl",
                                   map_location=device,
                                   weights_only=True)
                    )
                    print(r"load model 5: D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_aug\PGNN\10K\gcnn_aug_1.pkl")

                    model = model.to(device)
                    model.eval()
                elif policy['name'] == 'gcnn_0.1_5k':
                    model = GNNPolicy(cl=False)

                    # case118
                    model.load_state_dict(
                        torch.load(r"D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_0.1_gasse\5K\gcnn_0.1_gasse.pkl",
                                   map_location=device,
                                   weights_only=True)
                    )
                    print(r"load model 6: D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_0.1_gasse\5K\gcnn_0.1_gasse.pkl")
                    model = model.to(device)
                    model.eval()
                elif policy['name'] == 'gcnn_huake_5k':
                    model = GNNPolicy(cl=False)

                    # case118
                    model.load_state_dict(
                        torch.load(r"D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_0.1_huake\5K\gcnn_0.1_huake.pkl",
                                   map_location=device,
                                   weights_only=True)
                    )
                    print(r"load model 7: D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_0.1_huake\5K\gcnn_0.1_huake.pkl")
                    model = model.to(device)
                    model.eval()
                elif policy['name'] == 'IGNN_NoAMILP_5k':
                    model = GNNPolicy_Pinfo(cl=True)

                    # case118
                    model.load_state_dict(
                        torch.load(r"D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_aug\NoAMILP\5K\gcnn_aug_1.pkl",
                                   map_location=device,
                                   weights_only=True)
                    )
                    print(r"load model 8: D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_aug\NoAMILP\5K\gcnn_aug_1.pkl")
                    model = model.to(device)
                    model.eval()
                elif policy['name'] == 'gcnn_aug_5k':
                    model = GNNPolicy_Pinfo(cl=True)

                    # case118
                    model.load_state_dict(
                        torch.load(r"D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_aug\PGNN\5K\gcnn_aug_1.pkl",
                                   map_location=device,
                                   weights_only=True)
                    )
                    print(r"load model 9: D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_aug\PGNN\5K\gcnn_aug_1.pkl")

                    model = model.to(device)
                    model.eval()
                elif policy['name'] == 'gcnn_0.1_80k':
                    model = GNNPolicy(cl=False)

                    # case118
                    model.load_state_dict(
                        torch.load(r"D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_0.1_gasse\80K\gcnn_0.1_gasse.pkl",
                                   map_location=device,
                                   weights_only=True)
                    )
                    print(r"load model 10: D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_0.1_gasse\80K\gcnn_0.1_gasse.pkl")
                    model = model.to(device)
                    model.eval()
                elif policy['name'] == 'gcnn_huake_80k':
                    model = GNNPolicy(cl=False)

                    # case118
                    model.load_state_dict(
                        torch.load(r"D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_0.1_huake\80K\gcnn_0.1_huake.pkl",
                                   map_location=device,
                                   weights_only=True)
                    )
                    print(r"load model 11: D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_0.1_huake\80K\gcnn_0.1_huake.pkl")
                    model = model.to(device)
                    model.eval()
                elif policy['name'] == 'IGNN_NoAMILP_80k':
                    model = GNNPolicy_Pinfo(cl=True)

                    # case118
                    model.load_state_dict(
                        torch.load(r"D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_aug\NoAMILP\80K\gcnn_aug_1.pkl",
                                   map_location=device,
                                   weights_only=True)
                    )
                    print(r"load model 12: D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_aug\NoAMILP\80K\gcnn_aug_1.pkl")
                    model = model.to(device)
                    model.eval()
                elif policy['name'] == 'gcnn_aug_80k':
                    model = GNNPolicy_Pinfo(cl=True)

                    # case118
                    model.load_state_dict(
                        torch.load(r"D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_aug\PGNN\80K\gcnn_aug_1.pkl",
                                   map_location=device,
                                   weights_only=True)
                    )
                    print(r"load model 13: D:\LiJiamigFile\CAMBranch-ljmdata\KIDA_log\case118\gcnn_aug\PGNN\80K\gcnn_aug_1.pkl")

                    model = model.to(device)
                    model.eval()
                else:
                    raise Exception(f"Unrecognized GNN policy {policy['name']}")
                loaded_models[policy['name']] = model

            policy['model'] = loaded_models[policy['name']]

    print("running SCIP...")
    fieldnames = [
        'policy',
        'seed',
        'obj',
        'dual',
        'instance',
        'nnodes',
        'nlps',
        'nlpit',
        'nbranch',
        'featime',
        'inftime',
        'dual_integral_1e4',
        'btime',
        'stime',
        'gap',
        'status',
        'ndomchgs',
        'ncutoffs',
        'walltime',
        'proctime',
    ]

    running_dir = f"D:/LiJiamigFile/CAMBranch-ljmdata/online_log/{args.problem}"
    with open(f"{running_dir}/{result_file}", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_time = 0.0
        total_gap = 0.0
        total_solve_num = 0

        for instance in instances:
            print(f"{instance['type']}: {instance['path']}...")

            nn_nodes, stime = 0, 0
            
            for policy in branching_policies:
                m = scip.Model()
                if policy['seed'] == 'no seed':
                    m.setIntParam('display/verblevel', 0)
                elif policy['seed'] == 'no seed,restart,cuts':
                    m.setIntParam('separating/maxrounds', 0)
                    m.setIntParam('display/verblevel', 0)
                    # m.setIntParam('display/freq', 1)
                    m.setIntParam('presolving/maxrestarts', 0)
                else:
                    if policy['name'] == 'relpscost':
                        utilities_v2.init_scip_params_relpscost(m, seed=policy['seed'])
                    else:
                        utilities_v2.init_scip_params_evaluate(m, seed=policy['seed'])
                # m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time, default
                m.setRealParam('limits/time', time_limit)
                m.setRealParam('limits/gap', gap_limit)
                m.readProblem(f"{instance['path']}")
                if policy['name'] == 'vanillafullstrong':
                    m.setBoolParam('branching/vanillafullstrong/donotbranch', True)
                Dir1 = os.path.dirname(instance['path'])
                caseName = args.problem
                unitlist = utilities_pinfo.getunitdata(Dir1 + "\\5-" + caseName + "-机组数据.csv")
                buslist, load = IEEE_g.getbusdata(Dir1 + "\\1-" + caseName + "-母线名称.csv",
                                                  Dir1 + "\\2-" + caseName + "-母线负荷.csv",
                                                  Dir1 + "\\3-" + caseName + "-系统负荷.csv")
                linelist = None
                Blocklineindex = None
                PTDF_dict = None
                if args.do_pscost == 'YES':
                    do_pscost = True
                else:
                    do_pscost = False
                seed_tmp = policy['seed']
                file_name_tmp1 = os.path.basename(instance['path'])
                file_name_tmp2 = os.path.splitext(file_name_tmp1)[0]
                online_sample_dir = args.online_sample_dir + f'/{args.problem}/{file_name_tmp2}_seed{seed_tmp}_train'
                train_freq = args.train_freq

                brancher = PolicyBranching(policy, unitlist, load, linelist, Blocklineindex, PTDF_dict, online_sample_dir, do_pscost, file_name_tmp2, train_freq)
                m.includeBranchrule(
                    branchrule=brancher,
                    name=f"{policy['type']}:{policy['name']}",
                    desc=f"Custom PySCIPOpt branching policy.",
                    priority=666666, maxdepth=-1, maxbounddist=1)
                walltime = time.perf_counter()
                proctime = time.process_time()

                m.optimize()

                walltime = time.perf_counter() - walltime
                proctime = time.process_time() - proctime
                stime = m.getSolvingTime()
                nnodes = m.getNNodes()
                nlps = m.getNLPs()
                nlpit = m.getNLPIterations()
                gap = m.getGap()
                status = m.getStatus()
                ndomchgs = brancher.ndomchgs
                ncutoffs = brancher.ncutoffs
                nbranch = brancher.branch_count
                featime = brancher.fea_time
                inftime = brancher.inf_time
                final_obj = m.getObjVal()
                branchtime = stime - brancher.branch_begin_time
                dual_integral = 0.0
                dual_integral_trip = 0.0


                # # ***Should the intermediate information of the solution process be output?
                # time_left = max(m.getParam("limits/time") - m.getSolvingTime(), 0)
                # dual_integral_final_branch = (m.getSolvingTime()-brancher.time_record_list[-1]) * (m.getDualbound()-brancher.dual_record_list[-1])
                # # dual_integral_final_branch = 0.0
                # dual_integral_left = time_left * (m.getDualbound()-brancher.dual_record_list[0])
                # dual_integral = brancher.dual_integral + dual_integral_final_branch + dual_integral_left
                # dual_integral_trip = dual_integral/1e4


                writer.writerow({
                    'policy': f"{policy['type']}:{policy['name']}",
                    'seed': policy['seed'],
                    'obj': round(final_obj,1),
                    'dual': round(m.getDualbound(), 1),
                    'instance': instance['path'],
                    'nnodes': nnodes,
                    'nlps': nlps,
                    'nlpit': nlpit,
                    'nbranch': nbranch,
                    'featime': round(featime,3),
                    'inftime': round(inftime,3),
                    'dual_integral_1e4': round(dual_integral_trip,6),
                    'btime': round(branchtime,3),
                    'stime': round(stime,3),
                    'gap': round(gap,6),
                    'status': status,
                    'ndomchgs': ndomchgs,
                    'ncutoffs': ncutoffs,
                    'walltime': round(walltime,3),
                    'proctime': round(proctime,3),
                })
                csvfile.flush()


                # # ***Should the intermediate information of the solution process be output?
                # brancher.time_record_list.append(m.getTotalTime())
                # brancher.primal_record_list.append(m.getPrimalbound())
                # brancher.dual_record_list.append(m.getDualbound())
                # brancher.relgap_record_list.append(m.getGap())
                # brancher.nnodes_record_list.append(m.getNNodes())
                # brancher.bvarname_list.append(0)
                # brancher.genid_record_list.append(0)
                # brancher.t_record_list.append(0)
                # brancher.dual_integral_list.append(dual_integral)
                # data = list(zip(
                #     brancher.time_record_list,
                #     brancher.primal_record_list,
                #     brancher.dual_record_list,
                #     brancher.relgap_record_list,
                #     brancher.nnodes_record_list,
                #     brancher.bvarname_list,
                #     brancher.genid_record_list,
                #     brancher.t_record_list,
                #     brancher.dual_integral_list
                # ))
                # policyName = policy['name']
                # name = os.path.basename(os.path.dirname(instance['path']))
                # file_path = f"D:/LiJiamigFile/CAMBranch-ljmdata/online_log/{args.problem}/{name}-P-D-G-record-{policyName}.csv"
                # with open(file_path, 'w', newline='', encoding='utf-8') as f:
                #     writer_2 = csv.writer(f)
                #     writer_2.writerow(['time', 'primal', 'dual', 'rel gap', 'nodes', 'bvarname', 'genid', 't', 'dual_integral'])
                #     for row in data:
                #         formatted_row = [f"{x}" if isinstance(x, str) else f"{x:.6f}" for x in row]
                #         writer_2.writerow(formatted_row)
                # ## Output the current MILP solution log in the terminal window at once
                # # print(f"\ttime    primal      dual    \trel_gap   node_num    bvarname   genid    t    dual_integral")
                # # for row in data:
                # #     ti, pr, du, re, nn, bv, ge, tr, di= [f"{x}" if isinstance(x, str) else f"{x:.3f}" for x in row]
                # #     print(f"\t{ti}  {pr}  {du}  {re}  {nn}  {bv}  {ge}  {tr}  {di}")


                # # ***Should the intermediate information of the solution process be output?
                # # Output the result of the unit commitment
                # output_scuc_result_xlsx = f"D:/LiJiamigFile/CAMBranch-ljmdata/online_log/{args.problem}/{name}_Result_{policyName}.xlsx"
                # extractor = utilities_pinfo.SCIPResultExtractor()
                # output_scuc_result = extractor.extract(model=m, output_xlsx=output_scuc_result_xlsx)


                m.freeProb()
                print(f"  {policy['type']}:{policy['name']} {policy['seed']} - {nnodes} ({nnodes+2*(ndomchgs+ncutoffs)}) nodes {nlps} lps {stime:.2f} ({branchtime:.2f} bran {walltime:.2f} wall {proctime:.2f} proc) s. {status}")
                total_time += stime
                total_gap += gap
                total_solve_num += 1
        print(f"milp_num: {total_solve_num/len(seeds)}\navg_solve_time: {total_time / total_solve_num:.2f} s\navg_mipgap: {total_gap / total_solve_num:.2f}")
