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

MODEL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PolicyBranching(scip.Branchrule):

    def __init__(self, policy, unitlist, load, linelist, Blocklineindex, PTDF_dict, online_sample_dir, do_pscost, case_name, train_freq):
        super().__init__()

        self.policy_type = policy['type']
        self.policy_name = policy['name']

        # __是否输出各个.lp求解全过程的dualbound等变化信息
        self.time_record_list = []   # 求解器运行时间
        self.primal_record_list = []
        self.dual_record_list = []
        self.relgap_record_list = []
        self.nnodes_record_list = []
        self.bvarname_list = []
        self.genid_record_list = []
        self.t_record_list = []      # 分支变量的物理信息中的时段
        self.dual_integral_list = []  # dual_integral的历史
        self.dual_integral = 0.0


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
        self.pscost_eta = []
        # self.pscost_total_delta = []
        # self.pscost_zone = []
        self.gcnn_i_model = GNNPolicy(cl=True).to(MODEL_DEVICE)
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
        '''
        问: 在 __init__ 中定义 self.pinfo_buffer = {} 和 在 branchinitsol 中定义 self.pinfo_buffer = {}的区别?
        答: 在 __init__ 中定义 self.pinfo_buffer = {} 是“全局持久缓存”，在 branchinitsol 中定义是“单次求解局部缓存”。
            如果 __init__ 中初始化大对象，但某些策略根本没被启用（比如只做 relpscost 对比），就浪费了而 branchinitsol 是按需初始化
        例如: 
            model1 = create_model_A()
            branch_rule = PolicyBranching()
            model1.includeBranchrule(branch_rule)
            model1.optimize()   # 第一次求解：pinfo_buffer 存了 model1 的节点信息
            
            model2 = create_model_B()  # 完全不同的问题！
            model2.includeBranchrule(branch_rule)  # 同一个 branch_rule 实例！
            model2.optimize()   # 第二次求解：pinfo_buffer 仍包含 model1 的旧数据！
        解决办法: 每次读取新的milp问题,单独执行branch_rule = PolicyBranching()创建新的分支对象即可!
        '''

    def branchexeclp(self, allowaddcons):

        if self.branch_count == 0:
            self.branch_begin_time = self.model.getTotalTime()
        start_time = time.perf_counter()
        state = utilities_v2.extract_state(self.model, self.state_buffer)
        c_fea, e_fea, v_fea, norm_data = state
        if self.pinfo_buffer is None:
            varlist = self.model.getVars(transformed=True)
            varnamelist = [v.name for v in varlist]
            Pinfo_array = np.zeros((len(varnamelist), 2))
            assert v_fea['values'].shape[0] == len(varnamelist), 'gasse提取的LP内变量个数 ≠ pyscipopt获取的变量个数!'
            varId = utilities_pinfo.parse_candidate_vars(varnamelist)  # 获取所有变量的身份信息
            n_units = len(self.unitlist)  # 机组总数 P
            n_ts = len(self.load)  # 时段总数 Q
            pmax = np.array([u.pmax for u in self.unitlist])
            pmin = np.array([u.pmin for u in self.unitlist])
            # 定义整数变量类型集合（set 查询 O(1)）
            INTEGER_VAR_TYPES = {'uit', 'yit', 'zit', 'ycoldit'}
            mask = np.array(
                [var.name in INTEGER_VAR_TYPES for var in varId])  # (N,)，N表示当前LP内的变量总数,M表示当前LP内的二元整数变量总数
            gen_ids = np.array([var.genid - 1 for var in varId])  # (N,)，注意越界检查
            gen_ids_valid = gen_ids[mask]  # (M,)，存储二元变量的机组序号在unitlist中的索引
            t_ids = np.array([var.t - 1 for var in varId])  # (N,)，注意越界检查
            t_ids_valid = t_ids[mask]  # (M,)，存储二元变量的时段序号在load中的索引
            load_sums = np.array([self.load[t].LoadSum for t in t_ids_valid])  # (M,)，列表推导 + 转数组
            assert (gen_ids >= -1).all() and (
                    gen_ids < n_units).all(), f"genid {gen_ids} 超出 unitlist 范围 {n_units}!"
            assert (t_ids >= -1).all() and (t_ids < n_ts).all(), f"t {t_ids} 超出 load 范围 {n_ts}!"
            Pinfo_array[mask, 0] = pmax[gen_ids_valid] / load_sums
            Pinfo_array[mask, 1] = pmin[gen_ids_valid]
            self.pinfo_buffer = Pinfo_array

        v_fea_Pinfo = np.concatenate([v_fea['values'], self.pinfo_buffer], axis=1)  # (N, R+i)
        end_time = time.perf_counter()
        self.fea_time += (end_time - start_time)

        # SCIP internal branching rule
        if self.policy_type == 'internal':
            result = self.model.executeBranchRule(self.policy, allowaddcons)
            if policy['name'] == 'vanillafullstrong':
                assert result == scip.SCIP_RESULT.DIDNOTRUN
                cands, scores, npriocands, bestcand = self.model.getVanillafullstrongData()
                best_var = cands[bestcand]                                                       # 类型:pyscipopt.scip.variable
                best_var_Id = utilities_pinfo.parse_candidate_vars([best_var.name])       # 输入[best_var.name]列表,获取best_var的身份信息
                self.model.branchVar(best_var)
                result = scip.SCIP_RESULT.BRANCHED

            # __是否输出各个.lp求解全过程的dualbound等变化信息
            self.time_record_list.append(self.model.getTotalTime())
            self.primal_record_list.append(self.model.getPrimalbound())
            self.dual_record_list.append(self.model.getDualbound())
            if len(self.dual_record_list) > 1:
                self.dual_integral += (self.time_record_list[-1] - self.time_record_list[-2]) * (self.dual_record_list[-1]-self.dual_record_list[0])    # 累计dual_integral
                self.dual_integral_list.append(self.dual_integral)                                                                                      # 存储dual_integral的历史
            else:
                self.dual_integral_list.append(0.0)
            self.relgap_record_list.append(self.model.getGap())
            self.nnodes_record_list.append(self.model.getNNodes())
            if policy['name'] == 'vanillafullstrong':
                self.bvarname_list.append(best_var_Id[0].name)
                self.genid_record_list.append(best_var_Id[0].genid)
                self.t_record_list.append(best_var_Id[0].t)
            else:
                self.bvarname_list.append(0)
                self.genid_record_list.append(0)
                self.t_record_list.append(0)

        # custom policy branching
        else:
            candidate_vars, *_ = self.model.getPseudoBranchCands()
            candidate_mask = [var.getCol().getLPPos() for var in candidate_vars]

            if len(candidate_vars) == 1:
                best_var = candidate_vars[0]

            elif self.policy_type == 'gnn':
                start_time = time.perf_counter()
                GNNPolicy_Pinfo_list =  ['gcnn_aug', 'IGNN_NoAMILP', 'IGNN_5k', 'UGNN']
                if self.policy_name in GNNPolicy_Pinfo_list:
                    variable_new = v_fea_Pinfo
                else:
                    variable_new = v_fea['values']
                state_pt = (
                    torch.tensor(c_fea['values'], dtype=torch.float32).to(device=MODEL_DEVICE),  # float
                    torch.tensor(e_fea['indices'], dtype=torch.int64).to(device=MODEL_DEVICE),  # long
                    torch.tensor(e_fea['values'], dtype=torch.float32).to(device=MODEL_DEVICE),  # float
                    torch.tensor(variable_new, dtype=torch.float32).to(device=MODEL_DEVICE),  # float
                )
                with torch.no_grad():
                    var_logits = self.policy(*state_pt).cpu().numpy()

                candidate_scores = var_logits[candidate_mask]
                best_var = candidate_vars[candidate_scores.argmax()]
                end_time = time.perf_counter()
                self.inf_time += (end_time - start_time)

            else:
                raise NotImplementedError

            # __是否输出各个.lp求解全过程的dualbound等变化信息
            self.time_record_list.append(self.model.getTotalTime())
            self.primal_record_list.append(self.model.getPrimalbound())
            self.dual_record_list.append(self.model.getDualbound())
            if len(self.dual_record_list) > 1:
                self.dual_integral += (self.time_record_list[-1] - self.time_record_list[-2]) * (self.dual_record_list[-1]-self.dual_record_list[0])    # 累计dual_integral
                self.dual_integral_list.append(self.dual_integral)                                                                                      # 存储dual_integral的历史
            else:
                self.dual_integral_list.append(0.0)
            self.relgap_record_list.append(self.model.getGap())
            self.nnodes_record_list.append(self.model.getNNodes())
            best_var_Id = utilities_pinfo.parse_candidate_vars([best_var.name])  # 输入[best_var]列表,获取best_var的身份信息
            self.bvarname_list.append(best_var_Id[0].name)
            self.genid_record_list.append(best_var_Id[0].genid)
            self.t_record_list.append(best_var_Id[0].t)


            self.model.branchVar(best_var)
            result = scip.SCIP_RESULT.BRANCHED

        self.branch_count += 1  # 无论采用哪一种分支策略,只要执行,就加1

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
        choices=['gcnn', 'gcnn_0.1', 'gcnn_huake', 'IGNN_NoPgraph', 'IGNN_NoAMILP', 'IGNN_5k', 'gcnn_aug', 'UGNN']
    )
    parser.add_argument(
        '-w1',
        '--weightlist1',
        type=float,
        nargs='+',
        default=[1, 1e-4, 1e-4, 1e-4, 1e-1],
        help='Physical information score weighting coefficient vector, including: avgcost, hotstartcost, coldstartcost, initT, t.',
    )
    parser.add_argument(
        '-w2',
        '--weightlist2',
        type=float,
        nargs='+',
        default=[1, 0.5, 0.5, 0.1],
        help='variable type weight list, including: uit, yit, zit, ycoldit.',
    )
    parser.add_argument(
        '--sortlen', '-slen',
        help='sort len.',
        type=int,
        default=15,
    )
    parser.add_argument(
        '--phyInfoPartCount', '-pCount',
        help='physical informational calculation count.',
        type=int,
        default=100,
    )
    parser.add_argument('--alpha-cl', type=float, default=0.05)
    parser.add_argument('--alpha-reg', type=float, default=0.01)
    parser.add_argument(
        '--online_sample_dir',
        type=str,
        default = f'./online_data/samples'
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
    os.makedirs(args.online_sample_dir)
    instances = []
    seeds = [0, 2, 4]
    # seeds = ['no seed,restart,cuts']
    # seeds = ['no seed']
    internal_branchers = [] if args.internal_brancher is None else args.internal_brancher
    other_models = [] if args.ml_model is None else args.ml_model
    gnn_models = [] if args.gnn_model is None else args.gnn_model
    time_limit = 600.0
    gap_limit = 0.0000

    if len(args.prefix) == 0:
        args.prefix = '_'.join(internal_branchers + other_models + gnn_models)
    result_file = f"{args.problem}_{args.type}_{args.prefix}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    if args.problem == 'case118':

        #                    I1    I2    I3    I4    I5    I6    I7    I8    I9    I10
        user_selected_ids = [2404, 2405, 2433, 2434, 2435, 2501, 2504, 2505, 2532, 2534 ] 
        instances += [{'type': 'BESSs', 'path': f"./data/instances/case118/test_milp/case118_{i}/case118_{i}.lp"} for i in user_selected_ids]    # 共10个算例

    else:
        raise NotImplementedError

    branching_policies = []
    
    # GNN models
    for model in gnn_models:
        for seed in seeds:
            branching_policies.append({
                'type': 'gnn',  # 源码内容
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
    print(f"gpu: {args.gpu}")
    print(f"time limit: {time_limit} s")
    print(f'gap limit: {gap_limit:.4f} %')

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
                    model = GNNPolicy(cl=False).to(device)
                    # case118
                    model.load_state_dict(
                        torch.load(r"./model_log/case118\gcnn_0.1_gasse\0\case118-v2\10k\gcnn_0.1_gasse.pkl",
                                   weights_only=True)
                    )
                    print(r"load model 1: ./model_log/case118\gcnn_0.1_gasse\0\case118-v2\10k\gcnn_0.1_gasse.pkl")

                    model.eval()
                elif policy['name'] == 'gcnn_huake':
                    model = GNNPolicy(cl=False).to(device)

                    # case118
                    model.load_state_dict(
                        torch.load(r"./model_log/case118\gcnn_0.1_huake\0\case118-v2\10k\gcnn_0.1_huake.pkl", weights_only=True)
                    )
                    print(r"load model 2: ./model_log/case118\gcnn_0.1_huake\0\case118-v2\10k\gcnn_0.1_huake.pkl")

                    model.eval()
                elif policy['name'] == 'IGNN_NoPgraph':
                    model = GNNPolicy(cl=True).to(device)

                    # case118
                    model.load_state_dict(
                        torch.load(r"./model_log/case118\gcnn_aug\0\0.05_0.01\case118-v2\NoPgraph\10K\gcnn_aug_1.pkl", weights_only=True)
                    )
                    print(r"load model 3: ./model_log/case118\gcnn_aug\0\0.05_0.01\case118-v2\NoPgraph\10K\gcnn_aug_1.pkl")

                    model.eval()
                elif policy['name'] == 'IGNN_NoAMILP':
                    model = GNNPolicy_Pinfo(cl=True).to(device)

                    # case118
                    model.load_state_dict(
                        torch.load(r"./model_log/case118\gcnn_aug\0\0.05_0.01\case118-v2\NoAMILP\10k\gcnn_aug_1.pkl", weights_only=True)
                    )
                    print(r"load model 4: ./model_log/case118\gcnn_aug\0\0.05_0.01\case118-v2\NoAMILP\10k\gcnn_aug_1.pkl")

                    model.eval()
                elif policy['name'] == 'gcnn_aug':
                    model = GNNPolicy_Pinfo(cl=True).to(device)

                    # case118
                    model.load_state_dict(
                        torch.load(r"./model_log/case118\gcnn_aug\0\0.05_0.01\case118-v2\PGNN\10k\gcnn_aug_1.pkl", weights_only=True)
                    )
                    print(r"load model 6: ./model_log/case118\gcnn_aug\0\0.05_0.01\case118-v2\PGNN\10k\gcnn_aug_1.pkl")


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
                    m.setIntParam('presolving/maxrestarts', 0)
                else:
                    if policy['name'] == 'relpscost':
                        utilities_v2.init_scip_params_relpscost(m, seed=policy['seed'])
                    else:
                        utilities_v2.init_scip_params_evaluate(m, seed=policy['seed'])
                m.setRealParam('limits/time', time_limit)
                m.setRealParam('limits/gap', gap_limit)
                m.readProblem(f"{instance['path']}")
                if policy['name'] == 'vanillafullstrong':
                    m.setBoolParam('branching/vanillafullstrong/donotbranch', True)
                Dir1 = os.path.dirname(instance['path'])
                caseName = args.problem
                unitlist = utilities_pinfo.getunitdata(Dir1 + "\\5-" + caseName + "-unitdata.csv")
                buslist, load = IEEE_g.getbusdata(Dir1 + "\\1-" + caseName + "-busname.csv",
                                                  Dir1 + "\\2-" + caseName + "-busload.csv",
                                                  Dir1 + "\\3-" + caseName + "-systemload.csv")
                linelist = None
                Blocklineindex = None
                PTDF_dict = None

                if args.do_pscost == 'YES':
                    do_pscost = True
                else:
                    do_pscost = False

                seed_tmp = policy['seed']
                file_name_tmp1 = os.path.basename(instance['path'])    # 获取包含后缀名的文件名
                file_name_tmp2 = os.path.splitext(file_name_tmp1)[0]   # 获取纯文件名
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

                walltime = time.perf_counter() - walltime  # 总时间t1
                proctime = time.process_time() - proctime  # 进程时间t2

                stime = m.getSolvingTime()                 # SCIP记录的求解器时间t3, 关系: t1 >=t2, t1>=t3, t2和t3大小关系不确定
                nnodes = m.getNNodes()                     # SCIP记录的创建的节点总数,表示从根节点出发，SCIP 创建了多少个子节点（包括被剪掉的、求解过的、未处理的等）
                nlps = m.getNLPs()                         # SCIP记录的解决过的LP问题总数
                nlpit = m.getNLPIterations()               # SCIP记录的LP迭代总次数
                gap = m.getGap()
                status = m.getStatus()
                ndomchgs = brancher.ndomchgs               # SCIP记录的 变量域变化次数
                ncutoffs = brancher.ncutoffs               # SCIP记录的 剪枝次数
                nbranch = brancher.branch_count            # SCIP记录的 分支总数  # 2026.3.9新增
                featime = brancher.fea_time                # SCIP记录的 采样总时间
                inftime = brancher.inf_time                # SCIP记录的 推理总时间
                final_obj = m.getObjVal()                  # SCIP记录的 最终目标函数值

                dual_integral = 0.0
                dual_integral_trip = 0.0
                # __是否输出各个.lp求解全过程的dualbound等变化信息
                # time_left = max(m.getParam("limits/time") - m.getSolvingTime(), 0)
                # # 分支插件PolicyBranching里统计的永远是上一次分支的抬升面积,故最后1次分支的抬升只能在求解完毕后才能统计
                # dual_integral_final_branch = (m.getSolvingTime()-brancher.time_record_list[-1]) * (m.getDualbound()-brancher.dual_record_list[-1])
                # # dual_integral_final_branch = 0.0
                # # 假如提前到达gap=0,剩余的时间也要补充dual_integral,否则对这类优秀分支策略的评估不公平!
                # dual_integral_left = time_left * (m.getDualbound()-brancher.dual_record_list[0])
                # dual_integral = brancher.dual_integral + dual_integral_final_branch + dual_integral_left    # SCIP记录的 累计dual_integral
                # dual_integral_trip = dual_integral/1e4

                branchtime = stime - brancher.branch_begin_time

                writer.writerow({
                    'policy': f"{policy['type']}:{policy['name']}",
                    'seed': policy['seed'],
                    'obj': round(final_obj,1),
                    'instance': instance['path'],
                    'nnodes': nnodes,
                    'nlps': nlps,
                    'nlpit': nlpit,
                    'nbranch': nbranch,
                    'featime': round(featime,3),
                    'inftime': round(inftime,3),
                    'dual_integral_1e4': round(dual_integral_trip,6),  # 存储dual_integral的历史
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

                # __是否输出各个.lp求解全过程的dualbound等变化信息
                brancher.time_record_list.append(m.getTotalTime())
                brancher.primal_record_list.append(m.getPrimalbound())
                brancher.dual_record_list.append(m.getDualbound())
                brancher.relgap_record_list.append(m.getGap())
                brancher.nnodes_record_list.append(m.getNNodes())
                brancher.bvarname_list.append(0)      # 当求解器达到限制或最优,分支变量已经不存在,用0代替
                brancher.genid_record_list.append(0)  # 当求解器达到限制或最优,分支变量已经不存在,用0代替
                brancher.t_record_list.append(0)      # 当求解器达到限制或最优,分支变量已经不存在,用0代替
                brancher.dual_integral_list.append(dual_integral) # 存储dual_integral的历史
                data = list(zip(
                    brancher.time_record_list,
                    brancher.primal_record_list,
                    brancher.dual_record_list,
                    brancher.relgap_record_list,
                    brancher.nnodes_record_list,
                    brancher.bvarname_list,
                    brancher.genid_record_list,
                    brancher.t_record_list,
                    brancher.dual_integral_list  # 存储dual_integral的历史
                ))

                policyName = policy['name']
                name = os.path.basename(os.path.dirname(instance['path']))
                file_path = f"./online_log/{args.problem}/{name}-P-D-G-record-{policyName}.csv"
                # 写入 CSV 文件
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer_2 = csv.writer(f)
                    writer_2.writerow(['time', 'primal', 'dual', 'rel gap', 'nodes', 'bvarname', 'genid', 't', 'dual_integral'])  # dual_integral的历史
                    # 写入数据，保留 6 位小数
                    for row in data:
                        formatted_row = [f"{x}" if isinstance(x, str) else f"{x:.6f}" for x in row]
                        writer_2.writerow(formatted_row)

                # 输出机组组合结果
                # output_scuc_result_xlsx = f"./online_log/{args.problem}/{name}_Result_{policyName}.xlsx"
                # extractor = utilities_pinfo.SCIPResultExtractor()
                # output_scuc_result = extractor.extract(model=m, output_xlsx=output_scuc_result_xlsx)

                # 释放模型
                m.freeProb()
                # 每求解1个MILP实例结束后打印信息
                #          分支策略类型      分支策略名称        随机种子数值    SCIP创建的节点总数   节点代价经验公式=创建节点数+2*变量域改变次数+2*剪枝次数
                print(f"  {policy['type']}:{policy['name']} {policy['seed']} - {nnodes} ({nnodes+2*(ndomchgs+ncutoffs)}) nodes {nlps} lps {stime:.2f} ({branchtime:.2f} bran {walltime:.2f} wall {proctime:.2f} proc) s. {status}")

                total_time += stime
                total_gap += gap
                total_solve_num += 1

        print(f"milp_num: {total_solve_num/len(seeds)}\navg_solve_time: {total_time / total_solve_num:.2f} s\navg_mipgap: {total_gap / total_solve_num:.2f}")
