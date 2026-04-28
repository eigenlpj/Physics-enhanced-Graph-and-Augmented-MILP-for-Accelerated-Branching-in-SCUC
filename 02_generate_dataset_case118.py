"""
Adapted from ds4dm/learn2branch (https://github.com/ds4dm/learn2branch/blob/master/02_generate_dataset.py)
Modified for PGNN (Physics-enhanced Graph for SCUC) under the same MIT License.
"""
import os
import sys
import argparse
import multiprocessing as mp
import pickle
import glob
import numpy as np
import shutil
import gzip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.cluster.hierarchy import weighted

# 自定义库
import pyscipopt as scip  # scip10.0.0
import utilities_v2
import utilities_pinfo
import IEEE_g
sys.path.insert(0, os.path.abspath(f'models'))
from models.model import GNNPolicy

# 全局变量
MODEL_DEVICE='cuda' if torch.cuda.is_available() else 'cpu'  # 神经网络推理设备

class SamplingAgent(scip.Branchrule):

    def __init__(self, episode, instance, seed, out_queue, exploration_policy, query_expert_prob, out_dir,
                 unitlist, load,  case_name, gcnn_seed, iter_index, fullstrong_num,
                 linelist=None, Blocklineindex=None, PTDF_dict=None,follow_expert=True):
        self.episode = episode
        self.instance = instance
        self.seed = seed  # SCIP求解器的seed
        self.out_queue = out_queue
        self.exploration_policy = exploration_policy
        self.query_expert_prob = query_expert_prob
        self.out_dir = out_dir
        self.follow_expert = follow_expert

        self.rng = np.random.RandomState(seed)
        self.new_node = True
        self.sample_counter = 0

        self.unitlist = unitlist
        self.load = load
        self.linelist = linelist
        self.Blocklineindex = Blocklineindex
        self.PTDF_dict = PTDF_dict

        self.initial_time = 0.0
        self.current_time = 0.0
        self.delta_time = 0.0
        self.initial_dual = -float('inf')
        self.current_dual = -float('inf')
        self.delta_dual = 0.0
        self.total_delta_dual = 0.0

        self.branch_count = 0  # 分支次数 ≠ 分支样本个数sample_counter
        self.pscost_eta = []   # 整数变量的已分支次数
        self.pscost_total_delta = []  # 整数变量的对偶下界抬升累计值
        self.gcnn_acc_model = GNNPolicy(cl=True).to(MODEL_DEVICE)  # 初始化 GCNN
        self.old_model_file = None  # 上一次加载的模型文件的序号,例如:"D:/.../gcnn_aug_2.pkl"的序号为2
        self.case_name = case_name  # 与05_evaluate_online_gcnn.py的case_name不同,此处表示系统的名称,例如:case118,而不是文件的名称
        self.gcnn_seed = gcnn_seed  # 神经网络训练使用的seed

        self.iter_index = iter_index
        self.fullstrong_num = fullstrong_num

    def branchinit(self):
        self.khalil_root_buffer = {}  # 作用范围: 每个 B&B 树的根节点（每次 restart）
        self.state_buffer = {}

    def branchexeclp(self, allowaddcons):

        # === 1. 提取当前状态 ===
        state = utilities_v2.extract_state(self.model,self.state_buffer)
        c_fea, e_fea, v_fea, norm_data = state
        varlist = self.model.getVars(transformed=True)  #
        varnamelist = [v.name for v in varlist]
        Pinfo_array = np.zeros((len(varnamelist), 11))  # 创建一个 len(varnamelist) * 26 的numpy数组,存放各个变量的物理信息
        assert v_fea['values'].shape[0] == len(varnamelist), 'gasse提取的LP内变量个数 ≠ pyscipopt获取的变量个数!'
        varId = utilities_pinfo.parse_candidate_vars(varnamelist)  # 获取所有变量的身份信息
        n_units = len(self.unitlist)  # 机组总数 P
        n_ts = len(self.load)  # 时段总数 Q
        pmax = np.array([u.pmax for u in self.unitlist])
        pmin = np.array([u.pmin for u in self.unitlist])
        total_iniP = sum([u.iniP for u in self.unitlist])  # 计算该milp的所有火电机组的iniP之和
        keepT = np.array([0 for u in self.unitlist])       # 舍弃该信息,用0填充
        ramp = np.array([u.RU for u in self.unitlist])
        start_up = np.array([u.SU for u in self.unitlist])
        shut_down = np.array([u.SD for u in self.unitlist])
        keep_down = np.array([u.keepOffT for u in self.unitlist])  # ini起仍在最小停机时间内的时段数
        keep_up = np.array([u.keepOnT for u in self.unitlist])     # ini起仍在最小开机时间内的时段数
        # 定义整数变量类型集合（set 查询 O(1)）
        INTEGER_VAR_TYPES = {'uit', 'yit', 'zit', 'ycoldit'}
        # 向量化填充：只对符合条件的变量赋值
        mask = np.array([var.name in INTEGER_VAR_TYPES for var in varId])  # (N,)，N表示当前LP内的变量总数,M表示当前LP内的二元整数变量总数
        gen_ids = np.array([var.genid - 1 for var in varId])  # (N,)，注意越界检查
        gen_ids_valid = gen_ids[mask]  # (M,)，存储二元变量的机组序号在unitlist中的索引
        t_ids = np.array([var.t - 1 for var in varId])  # (N,)，注意越界检查
        t_ids_valid = t_ids[mask]  # (M,)，存储二元变量的时段序号在load中的索引
        load_sums = np.array([self.load[t].LoadSum for t in t_ids_valid])      # (M,)，列表推导 + 转数组

        load_sums_delta_tmp1 = np.array([item.LoadSum for item in self.load])  # (t_num,)，其中t_num=24或者96,24时段/96时段的系统负荷值
        load_sums_delta_tmp2 = np.concatenate(([load_sums_delta_tmp1[0] - total_iniP], np.diff(load_sums_delta_tmp1)))  # (t_num,)，每个时段的系统负荷与前一时段的系统负荷的差值.
        load_sums_delta_tmp3 = np.abs(load_sums_delta_tmp2)  # (M,)，取绝对值
        eps_tmp = 1e-8
        load_sums_delta_tmp4 = np.where(load_sums_delta_tmp3 == 0, eps_tmp, load_sums_delta_tmp3)  # 防止出现零,后续作为分母报错
        load_sums_delta = load_sums_delta_tmp4[t_ids_valid]  # (M,)，各二元变量的时段对应的系统负荷变化量

        # 安全检查
        assert (gen_ids >= -1).all() and (gen_ids < n_units).all(), f"genid {gen_ids} 超出 unitlist 范围 {n_units}!"
        assert (t_ids >= -1).all() and (t_ids < n_ts).all(), f"t {t_ids} 超出 load 范围 {n_ts}!"

        Pinfo_array[mask, 0] = pmax[gen_ids_valid] / load_sums
        Pinfo_array[mask, 1] = pmin[gen_ids_valid] / load_sums
        Pinfo_array[mask, 2] = keepT[gen_ids_valid]
        ramp_values = ramp[gen_ids_valid] / load_sums_delta
        ## 假如前后时段系统负荷变化比机组爬坡功率还小,则将计算得到的"大数"改为"1.0",表示此时段各机组的"爬坡调节性能都是均等的"
        Pinfo_array[mask, 3] = np.where(ramp_values > 1.0, 1.0, ramp_values)
        # 获取当前有效变量对应的机组的 keep_up 和 keep_down 值
        # gen_ids_valid 是机组索引数组, keep_up/keep_down 是机组属性数组
        current_keep_up_vals = keep_up[gen_ids_valid]      # 形状 (M,), 获取每个整数变量对应的机组从初始时段起仍需开机的时段数
        current_keep_down_vals = keep_down[gen_ids_valid]  # 形状 (M,), 获取每个整数变量对应的机组从初始时段起仍需停机的时段数
        # t_ids_valid 是当前变量对应的时段索引 t (从0开始)
        # 逻辑: 如果 t < keep_up, 则为 1, 否则为 0
        # 注意: keep_up 表示"初始起仍在最小开机时间内的时段数"。
        # 例如: keep_up=3, 意味着 t=0, 1, 2 时机组处于强制开机状态。当 t=3 时，约束解除。
        # 所以条件是 t < keep_up
        col_24_mask = (t_ids_valid < current_keep_up_vals)  # 布尔数组 (M,)
        col_25_mask = (t_ids_valid < current_keep_down_vals)  # 布尔数组 (M,)
        # 将布尔值转换为浮点数 1.0 或 0.0 并赋值
        # 使用 .astype(float) 确保类型匹配，虽然 numpy 通常会自动处理，但显式转换更安全
        Pinfo_array[mask, 4] = col_24_mask.astype(float)  # 是否处于最小开机锁定区 (1=是, 0=否)
        Pinfo_array[mask, 5] = col_25_mask.astype(float)  # 是否处于最小停机锁定区 (1=是, 0=否)
        Pinfo_array[mask, 6] = pmax[gen_ids_valid]
        Pinfo_array[mask, 7] = pmin[gen_ids_valid]
        Pinfo_array[mask, 8] = ramp[gen_ids_valid]
        Pinfo_array[mask, 9] = start_up[gen_ids_valid]
        Pinfo_array[mask, 10] = shut_down[gen_ids_valid]

        v_fea_Pinfo_pscost = np.concatenate([v_fea['values'], Pinfo_array], axis=1)  # (N, R+n_pinfo)   # R表示gasse源码提取的变量特征总列数,即19
        v_fea_dict = {
            'names': v_fea['names'],
            'values': v_fea_Pinfo_pscost, }
        state_new = c_fea, e_fea, v_fea_dict, norm_data

        # === 2. 执行对应分支策略 ===
        """
        函数说明:
        1)getPseudoBranchCands()
          Function: Gets branching candidates for pseudo solution branching (non-fixed variables)
          Returns:
          list of Variable – list of variables of pseudo branching candidates
          int – number of pseudo branching candidates
          int – number of candidates with maximal priority
        2)getLPBranchCands()
          Function: Gets branching candidates for pseudo solution branching (non-fixed variables)
          Returns:
          list of Variable – list of variables of LP branching candidates
          list of float – list of LP candidate solution values
          list of float – list of LP candidate fractionalities
          int – number of LP branching candidates
          int – number of candidates with maximal priority
          int – number of fractional implicit integer variables
        """
        # cands, *_ = self.model.getPseudoBranchCands()  # 是否考虑整数LP解作为分支候选变量, 是: .getPseudoBranchCands(), 否: getLPBranchCands()
        cands, *_ = self.model.getLPBranchCands()
        action_set = [c.getCol().getLPPos() for c in cands]
        query_expert = True  # 初始化为True

        query_expert = self.rng.rand() < self.query_expert_prob
        # query_expert = self.rng.rand() < 2.0  # ljm debug
        if query_expert:  # 强分支
            result = self.model.executeBranchRule('vanillafullstrong', allowaddcons)
            # result = self.model.executeBranchRule('relpscost', allowaddcons)
            # print(f"branch result: {result}")  # ljm debug
            cands_, scores, npriocands, bestcand = self.model.getVanillafullstrongData()
            # if scores is None:  # ljm debug
            #     print(f"len_scores: None")
            # else:
            #     print(f"len_scores: {len(scores)}")
            assert result == scip.SCIP_RESULT.DIDNOTRUN
            assert all([c1.getCol().getLPPos() == c2.getCol().getLPPos() for c1, c2 in zip(cands, cands_)])
            expert_action = action_set[bestcand]   # 最佳候选变量的全局索引
            self.model.branchVar(cands[bestcand])  # 执行分支
            result = scip.SCIP_RESULT.BRANCHED
        else:  # pscost分支,不记录样本
            # result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)
            result = scip.SCIP_RESULT.DIDNOTFIND
            best_var = cands[0]
            expert_action = action_set[-1]
            scores = [-1]  # 标记为无效样本

        data = [state_new, {'khalil_info': [-1]}, expert_action, action_set, scores]

        # === 4. 获取reward ===
        # 以对偶下界提升值作为奖励
        if self.sample_counter == 0:
            self.initial_time = self.model.getSolvingTime()  # 仅在第1次有效分支前记录initial solving time
            self.initial_dual = self.model.getDualbound()    # 仅在第1次有效分支前记录initial dual bound
            self.delta_time = 0.0  # 第i次分支->第i+1次分支的时间在第i+1次分支才能观察到
            self.delta_dual = 0.0  # 第i次分支的对偶下界值要在第i+1次分支才能观察到,因此我们将第1次分支的delta_dual记录为0.0,表示第1次分支之前不记录对偶下界提升
        else:
            self.delta_time = self.model.getSolvingTime() - self.current_time  # 上一次分支到现在过去的时间
            self.delta_dual = self.model.getDualbound() - self.current_dual    # 当前分支的对偶下界减去上一次分支时的对偶下界,得到上一次分支的对偶下界提升值
        self.current_time = self.model.getSolvingTime()
        self.current_dual = self.model.getDualbound()  # 更新当前分支的对偶下界
        self.total_delta_dual += self.delta_dual       # 累计的对偶下界提升值,用于rl训练

        # === 5. 存储有效的分支样本===
        # Do not record inconsistent scores. May happen if SCIP was early stopped (time limit).
        if scores is not None and len(scores) > 1 and (not query_expert or all(s > 0 for s in scores)):
        # 条件1: 评分列表长度大于1; 条件2: 不查询vanillafullstrong; 条件3: 分数均为正 -> 同时满足条件1和2,或者 同时满足条件1和3
            filename = f'{self.out_dir}/sample_{self.episode}_{self.sample_counter}.pkl'
            with gzip.open(filename, 'wb') as f:
                pickle.dump({
                    'episode': self.episode,
                    'instance': self.instance,
                    'seed': self.seed,
                    'node_number': self.model.getCurrentNode().getNumber(),
                    'node_depth': self.model.getCurrentNode().getDepth(),
                    'data': data,                               # 第i次分支的节点状态
                    'var_names': varnamelist,                   # 当前LP所有变量名列表，用于后续解析expert_action对应的具体物理变量
                    'delta_time': self.delta_time,              # 第i-1次分支到第i次分支经过的时间
                    'query_expert_prob':self.query_expert_prob,
                    'delta_dual': self.delta_dual,              # 获取MC回报: 第 i-1 次强分支的对偶下界抬升
                    'acion_seq': self.sample_counter,           # 获取MC回报: 第 i 次强分支的分支序号i
                    'total_delta_dual': self.total_delta_dual,  # RL的奖励

                    }, f)

            self.out_queue.put({
                'type': 'sample',
                'episode': self.episode,
                'instance': self.instance,
                'seed': self.seed,
                'node_number': self.model.getCurrentNode().getNumber(),
                'node_depth': self.model.getCurrentNode().getDepth(),
                'filename': filename,
            })

            self.sample_counter += 1  # 生成有效分支样本后才能加1,无效样本不能加1

        self.branch_count += 1  # 无论采用哪一种分支策略,只要执行,就加1

        return {"result": result}

    # scip10.0.0由于某些原因调用如下两个分支的基本方法时,将SamplingAgent分支舍弃,回归至求解器默认分支
    def branchexecext(self, allowaddcons):
        return {"result": scip.SCIP_RESULT.DIDNOTFIND}

    def branchexecps(self, allowaddcons):
        return {"result": scip.SCIP_RESULT.DIDNOTFIND}

def make_samples(in_queue, out_queue):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """

    while True:
        (episode, instance, seed, exploration_policy, query_expert_prob, time_limit, out_dir,
         out_dir2, logfile, iter_index, fullstrong_num) = in_queue.get()
        milpname = os.path.splitext(os.path.basename(instance))[0]

        Dir1 = os.path.dirname(instance)
        caseName = milpname.split('_')[0]                                                     # 例如: case118
        unitlist = utilities_pinfo.getunitdata(Dir1 + "\\5-" + caseName + "-机组数据.csv")
        buslist, load = IEEE_g.getbusdata(Dir1 + "\\1-" + caseName + "-母线名称.csv",
                                   Dir1 + "\\2-" + caseName + "-母线负荷.csv",
                                   Dir1 + "\\3-" + caseName + "-系统负荷.csv")

        # slack_bus = 66  # case118的平衡节点
        # PlinelimitCount, linelist = IEEE_g.getLinedata(Dir1 + "\\4-" + caseName + "-线路参数.csv", len(buslist), int(slack_bus), Dir1 + "\\PTDF_matrix.csv")
        # Blocklineindex = utilities_pinfo.getPlineTrueIndex(linelist, Dir1 + "\\4-" + caseName + "-阻塞线路参数.csv")
        # PTDF_dict = IEEE_g.getPTDFdata(Dir1 + "\\PTDF_matrix.csv")
        # # print(f"1号发电机信息:\n{unitlist[0].pmax} MW\n1号母线负荷信息:\n{buslist[0].LoadP} MW\n1号系统负荷信息:\n{load[0].LoadSum} MW\n1号线路信息:\n{linelist[0].Pmax}\n")
        # # print(f"尾号发电机信息:\n{unitlist[-1].pmax} MW\n尾号母线负荷信息:\n{buslist[-1].LoadP}MW\n尾号系统负荷信息:\n{load[-1].LoadSum} MW\n尾号线路信息:\n{linelist[-1].Pmax}\n")
        # # print(f"阻塞线路序号:\n{Blocklineindex}\n")
        # # end_PTDF = len(linelist)-1
        # # print(f"PTDF矩阵[1,1]元素:\n{PTDF_dict['1'][1]}\nPTDF矩阵[1,end_PTDF]元素:\n{PTDF_dict['1'][end_PTDF]}\n")
        linelist = None
        Blocklineindex = None
        PTDF_dict = None

        m = scip.Model()
        utilities_v2.init_scip_params_collect(m, seed=seed)
        m.setIntParam('timing/clocktype', 2)
        m.readProblem(f'{instance}')

        time_limit_final = time_limit
        m.setRealParam('limits/time', time_limit_final)
        # gap_limit = 0.0001  # 默认全部算例都是0.01%的gap限制
        gap_limit = 0.0  # 默认全部算例都是0.0的gap限制
        m.setRealParam('limits/gap', gap_limit)

        # 日志记录
        limit_gap = m.getParam('limits/gap')
        root_cut_set = m.getParam('separating/maxroundsroot')
        node_cut_set = m.getParam('separating/maxrounds')
        restart_set = m.getParam('presolving/maxrestarts')
        print(f'[w {os.getpid()}] episode {episode}, seed {seed}, query_expert_prob {query_expert_prob}')
        utilities_v2.log(f"gapLimit {limit_gap * 100:.2f} %, rootCut {root_cut_set}, nodeCut {node_cut_set}, restart {restart_set}, "
                         f"processing milp: --{milpname}...", logfile)

        branchrule = SamplingAgent(
            episode=episode,
            instance=instance,
            seed=seed,
            out_queue=out_queue,
            exploration_policy=exploration_policy,
            query_expert_prob=query_expert_prob,
            out_dir = out_dir,
            unitlist= unitlist,
            load= load,
            case_name=caseName,
            gcnn_seed=0,
            iter_index=iter_index,
            fullstrong_num=fullstrong_num,
            linelist=linelist,
            Blocklineindex=Blocklineindex,
            PTDF_dict=PTDF_dict,
            follow_expert= True
            )

        m.includeBranchrule(
            branchrule=branchrule,
            name="Sampling branching rule", desc="",
            priority=666666, maxdepth=-1, maxbounddist=1)

        m.setBoolParam('branching/vanillafullstrong/integralcands', False)
        m.setBoolParam('branching/vanillafullstrong/scoreall', True)
        m.setBoolParam('branching/vanillafullstrong/collectscores', True)
        m.setBoolParam('branching/vanillafullstrong/donotbranch', True)
        m.setBoolParam('branching/vanillafullstrong/idempotent', True)

        out_queue.put({
            'type': 'start',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })

        m.optimize()
        f_gap = m.getGap()
        m.freeProb()

        print(f"[w {os.getpid()}] episode {episode} done, fin_gap: {f_gap * 100:.2f} %, {branchrule.sample_counter} samples/ {branchrule.branch_count} bran, "
              f"ini/fin_dual: {branchrule.initial_dual:.2f}/{branchrule.current_dual:.2f}, delta_dual: {branchrule.total_delta_dual:.2f}")

        out_queue.put({
            'type': 'done',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })


def send_orders(orders_queue, instances, seed, exploration_policy, query_expert_prob:list, time_limit:list, out_dir,
                out_dir2, logfile, iter_index, fullstrong_num):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    exploration_policy : str
        Branching strategy for exploration.
    query_expert_prob : float in [0, 1]       # plan1: 只设置1个采样概率
                       list, float in [0, 1]  # plan2: 设置2个采样概率
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]   # plan1: 只设置1个采样时间
                list, float in [0, 1]  # plan2: 设置2个采样时间
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    out_dir2: str
        日志文件所在的文件夹路径
    logfile: str
        日志文件完整路径
    iter_index: int
        当前迭代次数
    fullstrong_num: int
        第1次迭代时强分支执行次数
    """
    rng = np.random.RandomState(seed)

    episode = 0

    while True:
        instance = rng.choice(instances)
        seed = rng.randint(0, 2 ** 31)
        if len(query_expert_prob) > 1:
            rng_prob_list = [0.5, 0.5]
            query_expert_prob_tmp = rng.choice(query_expert_prob, p=rng_prob_list)
        else:
            query_expert_prob_tmp = query_expert_prob[0]
        if query_expert_prob_tmp == query_expert_prob[-1]:
            time_limit_tmp = time_limit[-1]
        else:
            time_limit_tmp = time_limit[0]
        orders_queue.put([episode, instance, seed, exploration_policy, query_expert_prob_tmp, time_limit_tmp, out_dir,
                          out_dir2, logfile, iter_index, fullstrong_num])
        episode += 1

def collect_samples(instances, out_dir, rng, n_samples, n_jobs,
                        exploration_policy, query_expert_prob:list, time_limit:list, out_dir2, logfile,
                        iter_index, fullstrong_num):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert
    brancher.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    exploration_policy : str
        Exploration policy (branching rule) for sampling.
    query_expert_prob : float in [0, 1]       # plan1: CAMbranch源码
                       list, float in [0, 1]  # plan2: 预选随机概率
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]   # plan1: CAMbranch源码
                list, float in [0, 1]  # plan2: 预选随机概率
        Maximum running time for an episode, in seconds.
    out_dir2: str
        日志文件所在的文件夹路径
    logfile: str
        日志文件完整路径
    iter_index: int
        主程序02_generate-v5.py当前迭代次数
    fullstrong_num: int
        第1次迭代时强分支执行次数
    """
    os.makedirs(out_dir, exist_ok=True)

    # start workers
    orders_queue = mp.Queue(maxsize=2*n_jobs)
    answers_queue = mp.SimpleQueue()
    workers = []
    for i in range(n_jobs):
        p = mp.Process(
                target=make_samples,
                args=(orders_queue, answers_queue),
                daemon=True)
        workers.append(p)
        p.start()

    tmp_samples_dir = f'{out_dir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # start dispatcher
    dispatcher = mp.Process(
        target=send_orders,
        args=(orders_queue, instances, rng.randint(0, 2 ** 31), exploration_policy, query_expert_prob, time_limit, tmp_samples_dir,
              out_dir2, logfile, iter_index, fullstrong_num),
              daemon=True)
    dispatcher.start()

    # record answers and write samples
    buffer = {}
    current_episode = 0
    i = 0
    in_buffer = 0
    while i < n_samples:
        sample = answers_queue.get()

        # add received sample to buffer
        if sample['type'] == 'start':
            buffer[sample['episode']] = []
        else:
            buffer[sample['episode']].append(sample)
            if sample['type'] == 'sample':
                in_buffer += 1

        # if any, write samples from current episode
        while current_episode in buffer and buffer[current_episode]:
            samples_to_write = buffer[current_episode]
            buffer[current_episode] = []

            for sample in samples_to_write:

                # if no more samples here, move to next episode
                if sample['type'] == 'done':
                    del buffer[current_episode]
                    current_episode += 1

                # else write sample
                else:
                    os.rename(sample['filename'], f'{out_dir}/sample_{i+1}.pkl')
                    in_buffer -= 1
                    i += 1
                    print_step = max(1, n_samples // 100)  # 当总样本数>100,每100个打印1次;当总样本数不足100,每1个打印1次
                    if i % print_step == 0 and i > 0:
                        print(f"[m {os.getpid()}] {i} / {n_samples} samples written, ep {sample['episode']} ({in_buffer} in buffer).")

                    # early stop dispatcher (hard)
                    if in_buffer + i >= n_samples and dispatcher.is_alive():
                        dispatcher.terminate()
                        print(f"[m {os.getpid()}] dispatcher stopped...")

                    # as soon as enough samples are collected, stop
                    if i == n_samples:
                        buffer = {}
                        break

    # stop all workers (hard)
    for p in workers:
        p.terminate()

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',

        choices=['case118', 'case300', 'case1888', 'case2383', 'case2848', '24GX', 'case2869'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities_v2.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--train_size',
        help='Number of train samples.',
        type=int,
        default=1000,
    )
    parser.add_argument(
        '--valid_size',
        help='Number of valid samples.',
        type=int,
        default=200,
    )
    # parser.add_argument(
    #     '--test_size',
    #     help='Number of test samples.',
    #     type=int,
    #     default=200,
    # )
    parser.add_argument(
        '--time_limit',
        nargs='+',
        type=float,
        choices=[3600.0, 7200.0, 10800.0, 14400.0],
        help="The maximal time of SCIP solver in seconds to run."
    )

    parser.add_argument(
        '--iter',
        help='Number of current iteration.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '-f_min','--fullstrong_num_min',
        help='minimum of fullstrong branch number.',
        type=int,
        default=5,
    )
    parser.add_argument(
        '-f_max','--fullstrong_num_max',
        help='maximum of fullstrong branch number.',
        type=int,
        default=20,
    )
    args = parser.parse_args()

    print(f"seed {args.seed}")

    train_size = args.train_size
    valid_size = args.valid_size
    exploration_strategy = 'pscost'
    node_record_prob = [0.05]
    time_limit = args.time_limit

    # 其他相关路径,请根据不同的case进行修改!
    base_dir = r"./data/instances/case118"                         # instances所在文件夹路径 case118
    finished_dir = r"./data/instances/case118/case118_finished_milp.csv"
    train_milp_begin = 1       # case118,  case2383 , 24GX, case1888
    train_milp_end = 2001
    valid_milp_begin = 2001
    valid_milp_end = 2401
    test_milp_begin = 2401
    test_milp_end = 2601

    # 样本保存路径,日志文件路径
    out_dir = f'./samples/{args.problem}_iter{args.iter}'   # 请手动修改样本的具体生成路径,为提高程序加载效率,不建议设置在当前py文件所在的文件夹！
    # create output directory, throws an error if it already exists
    os.makedirs(out_dir)
    logfile = os.path.join(out_dir,f'{args.problem}-collect-samples-log.txt')                         # 记录当前collect samples过程处理过的milp文件名称
    utilities_v2.log(f"MILP instance dir: {base_dir}", logfile)

    # 寻找有效的milp路径,支持断点续跑(需要事先使用logfile手动填写caseXXX_finished_milp.csv的信息)
    instances_train = []
    instances_valid = []
    instances_test = []
    for i in range(train_milp_begin, train_milp_end):  # 包括 1 到 xxx
        folder_name = f"train_milp/{args.problem}_{i}"
        file_path = os.path.join(base_dir, folder_name, f"{args.problem}_{i}.lp")
        done_file_list = utilities_pinfo.load_done_milp_paths(os.path.join(base_dir, 'train_milp'), finished_dir)
        if os.path.isfile(file_path) and f'{args.problem}_{i}' not in done_file_list:
            instances_train.append(file_path)
    for i in range(valid_milp_begin, valid_milp_end):  # 包括 xxx+1 到 yyy
        folder_name = f"valid_milp/{args.problem}_{i}"
        file_path = os.path.join(base_dir, folder_name, f"{args.problem}_{i}.lp")
        done_file_list = utilities_pinfo.load_done_milp_paths(os.path.join(base_dir, 'valid_milp'), finished_dir)
        if os.path.isfile(file_path) and f'{args.problem}_{i}' not in done_file_list:
            instances_valid.append(file_path)
    for i in range(test_milp_begin, test_milp_end):    # 包括 yyy+1 到 zzz
        folder_name = f"test_milp/{args.problem}_{i}"
        file_path = os.path.join(base_dir, folder_name, f"{args.problem}_{i}.lp")
        done_file_list = utilities_pinfo.load_done_milp_paths(os.path.join(base_dir, 'test_milp'), finished_dir)
        if os.path.isfile(file_path) and f'{args.problem}_{i}' not in done_file_list:
            instances_test.append(file_path)

    utilities_v2.log(f"{len(instances_train)} train instances for {train_size} samples", logfile)
    utilities_v2.log(f"{len(instances_valid)} validation instances for {valid_size} samples", logfile)
    utilities_v2.log(f"time_limit[CAMbranch, FSB]: {time_limit}", logfile)


 # 生成训练集
    rng = np.random.RandomState(args.seed)
    collect_samples(instances_train, out_dir + '/train', rng, train_size,
                    args.njobs, exploration_policy=exploration_strategy,
                    query_expert_prob=node_record_prob,
                    time_limit=time_limit, out_dir2=out_dir, logfile=logfile,
                    iter_index=args.iter, fullstrong_num=rng.randint(args.fullstrong_num_min, args.fullstrong_num_max)
                    )

 # 生成valid集
    if args.iter == 1:
        rng = np.random.RandomState(args.seed + 1)
        collect_samples(instances_valid, out_dir + '/valid', rng, valid_size,
                        args.njobs, exploration_policy=exploration_strategy,
                        query_expert_prob=node_record_prob,
                        time_limit=time_limit, out_dir2=out_dir, logfile=logfile,
                        iter_index=args.iter, fullstrong_num=rng.randint(args.fullstrong_num_min, args.fullstrong_num_max)
                        )

 # 生成test集
 #    rng = np.random.RandomState(args.seed + 2)
 #    collect_samples(instances_test, out_dir + '/test', rng, test_size,
 #                    args.njobs, exploration_policy=exploration_strategy,
 #                    query_expert_prob=node_record_prob,
 #                    time_limit = time_limit, out_dir2=out_dir, logfile=logfile,
 #                    iter_index = args.iter, fullstrong_num = args.fullstrong_num
 #                    )
