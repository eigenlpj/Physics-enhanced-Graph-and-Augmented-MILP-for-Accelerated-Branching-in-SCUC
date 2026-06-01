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

import pyscipopt as scip
import utilities_v2
import utilities_pinfo
import IEEE_g
sys.path.insert(0, os.path.abspath(f'models'))  
from models.model import GNNPolicy

MODEL_DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

class SamplingAgent(scip.Branchrule):

    def __init__(self, episode, instance, seed, out_queue, exploration_policy, query_expert_prob, out_dir,
                 unitlist, load,  case_name, gcnn_seed, iter_index, fullstrong_num,
                 linelist=None, Blocklineindex=None, PTDF_dict=None,follow_expert=True):      
        self.episode = episode
        self.instance = instance
        self.seed = seed  
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
        self.branch_count = 0  
        self.pscost_eta = [] 
        self.pscost_total_delta = []  
        self.gcnn_acc_model = GNNPolicy(cl=True).to(MODEL_DEVICE)  
        self.old_model_file = None  
        self.case_name = case_name  
        self.gcnn_seed = gcnn_seed  

        self.iter_index = iter_index
        self.fullstrong_num = fullstrong_num

    def branchinit(self):
        self.khalil_root_buffer = {}
        self.state_buffer = {}

    def branchexeclp(self, allowaddcons):

        state = utilities_v2.extract_state(self.model,self.state_buffer)
        c_fea, e_fea, v_fea, norm_data = state
        varlist = self.model.getVars(transformed=True)
        varnamelist = [v.name for v in varlist]
        Pinfo_array = np.zeros((len(varnamelist), 11))
        assert v_fea['values'].shape[0] == len(varnamelist), 'variables from gasse ≠ variables from pyscipopt!'
        varId = utilities_pinfo.parse_candidate_vars(varnamelist)
        n_units = len(self.unitlist)
        n_ts = len(self.load)
        pmax = np.array([u.pmax for u in self.unitlist])
        pmin = np.array([u.pmin for u in self.unitlist])
        total_iniP = sum([u.iniP for u in self.unitlist])
        keepT = np.array([0 for u in self.unitlist])
        ramp = np.array([u.RU for u in self.unitlist])
        start_up = np.array([u.SU for u in self.unitlist])
        shut_down = np.array([u.SD for u in self.unitlist])
        keep_down = np.array([u.keepOffT for u in self.unitlist])
        keep_up = np.array([u.keepOnT for u in self.unitlist])
        INTEGER_VAR_TYPES = {'uit', 'yit', 'zit', 'ycoldit'}
        mask = np.array([var.name in INTEGER_VAR_TYPES for var in varId])  # (N,)
        gen_ids = np.array([var.genid - 1 for var in varId])  # (N,)
        gen_ids_valid = gen_ids[mask]  # (M,)
        t_ids = np.array([var.t - 1 for var in varId])  # (N,)
        t_ids_valid = t_ids[mask]  # (M,)
        load_sums = np.array([self.load[t].LoadSum for t in t_ids_valid])      # (M,)

        load_sums_delta_tmp1 = np.array([item.LoadSum for item in self.load])  # (t_num,)
        load_sums_delta_tmp2 = np.concatenate(([load_sums_delta_tmp1[0] - total_iniP], np.diff(load_sums_delta_tmp1)))  # (t_num,)
        load_sums_delta_tmp3 = np.abs(load_sums_delta_tmp2)  # (M,)
        eps_tmp = 1e-8
        load_sums_delta_tmp4 = np.where(load_sums_delta_tmp3 == 0, eps_tmp, load_sums_delta_tmp3)
        load_sums_delta = load_sums_delta_tmp4[t_ids_valid]  # (M,)
        assert (gen_ids >= -1).all() and (gen_ids < n_units).all(), f"genid {gen_ids}  is out of the range of the unitlist {n_units}!"
        assert (t_ids >= -1).all() and (t_ids < n_ts).all(), f"t {t_ids}  is out of the range of load {n_ts}!"

        Pinfo_array[mask, 0] = pmax[gen_ids_valid] / load_sums
        Pinfo_array[mask, 1] = pmin[gen_ids_valid] / load_sums
        Pinfo_array[mask, 2] = keepT[gen_ids_valid]
        ramp_values = ramp[gen_ids_valid] / load_sums_delta
        Pinfo_array[mask, 3] = np.where(ramp_values > 1.0, 1.0, ramp_values)
        current_keep_up_vals = keep_up[gen_ids_valid]      # (M,)
        current_keep_down_vals = keep_down[gen_ids_valid]  # (M,)
        col_24_mask = (t_ids_valid < current_keep_up_vals)  #  (M,)
        col_25_mask = (t_ids_valid < current_keep_down_vals)  # (M,)
        Pinfo_array[mask, 4] = col_24_mask.astype(float)
        Pinfo_array[mask, 5] = col_25_mask.astype(float)
        Pinfo_array[mask, 6] = pmax[gen_ids_valid]
        Pinfo_array[mask, 7] = pmin[gen_ids_valid]
        Pinfo_array[mask, 8] = ramp[gen_ids_valid]
        Pinfo_array[mask, 9] = start_up[gen_ids_valid]
        Pinfo_array[mask, 10] = shut_down[gen_ids_valid]

        v_fea_Pinfo = np.concatenate([v_fea['values'], Pinfo_array], axis=1)  # (N, R+11)
        v_fea_dict = {
            'names': v_fea['names'],
            'values': v_fea_Pinfo, }
        state_new = c_fea, e_fea, v_fea_dict, norm_data

        ## Whether to consider integer LP solutions as candidate branching variables is:
        ## if .getPseudoBranchCands() is called; otherwise, getLPBranchCands() is called.
        # cands, *_ = self.model.getPseudoBranchCands()
        cands, *_ = self.model.getLPBranchCands()
        action_set = [c.getCol().getLPPos() for c in cands]

        query_expert = self.rng.rand() < self.query_expert_prob
        if query_expert:
            result = self.model.executeBranchRule('vanillafullstrong', allowaddcons)
            cands_, scores, npriocands, bestcand = self.model.getVanillafullstrongData()
            assert result == scip.SCIP_RESULT.DIDNOTRUN
            assert all([c1.getCol().getLPPos() == c2.getCol().getLPPos() for c1, c2 in zip(cands, cands_)])
            expert_action = action_set[bestcand]
            self.model.branchVar(cands[bestcand])
            result = scip.SCIP_RESULT.BRANCHED
        else:
            # result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)
            result = scip.SCIP_RESULT.DIDNOTFIND
            best_var = cands[0]
            expert_action = action_set[-1]
            scores = [-1]

        data = [state_new, {'khalil_info': [-1]}, expert_action, action_set, scores]

        if self.sample_counter == 0:
            self.initial_time = self.model.getSolvingTime()
            self.initial_dual = self.model.getDualbound()
            self.delta_time = 0.0
            self.delta_dual = 0.0
        else:
            self.delta_time = self.model.getSolvingTime() - self.current_time
            self.delta_dual = self.model.getDualbound() - self.current_dual
        self.current_time = self.model.getSolvingTime()
        self.current_dual = self.model.getDualbound()
        self.total_delta_dual += self.delta_dual

        # Do not record inconsistent scores. May happen if SCIP was early stopped (time limit).
        if scores is not None and len(scores) > 1 and (not query_expert or all(s > 0 for s in scores)):
        # Condition 1: The length of the score list is greater than 1;
        # Condition 2: Do not query "vanillafullstrong";
        # Condition 3: All scores are positive
        # -> Satisfy both Condition 1 and Condition 2, or both Condition 1 and Condition 3
            filename = f'{self.out_dir}/sample_{self.episode}_{self.sample_counter}.pkl'
            with gzip.open(filename, 'wb') as f:
                pickle.dump({
                    'episode': self.episode,
                    'instance': self.instance,
                    'seed': self.seed,
                    'node_number': self.model.getCurrentNode().getNumber(),
                    'node_depth': self.model.getCurrentNode().getDepth(),
                    'data': data,

                    'var_names': varnamelist,
                    'delta_time': self.delta_time,
                    'query_expert_prob':self.query_expert_prob,
                    'delta_dual': self.delta_dual,
                    'acion_seq': self.sample_counter,
                    'total_delta_dual': self.total_delta_dual,

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

            self.sample_counter += 1

        self.branch_count += 1

        return {"result": result}

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
        caseName = milpname.split('_')[0]
        unitlist = utilities_pinfo.getunitdata(Dir1 + "\\5-" + caseName + "-机组数据.csv")
        buslist, load = IEEE_g.getbusdata(Dir1 + "\\1-" + caseName + "-母线名称.csv",
                                   Dir1 + "\\2-" + caseName + "-母线负荷.csv",
                                   # Dir1 + "\\3-" + caseName + "-系统负荷+旋转备用+负荷曲线-24时段-算例1.csv")
                                   Dir1 + "\\3-" + caseName + "-系统负荷.csv")
        linelist = None
        Blocklineindex = None
        PTDF_dict = None

        m = scip.Model()
        utilities_v2.init_scip_params_collect(m, seed=seed)
        m.setIntParam('timing/clocktype', 2)
        m.readProblem(f'{instance}')
        time_limit_final = time_limit
        m.setRealParam('limits/time', time_limit_final)
        gap_limit = 0.0
        m.setRealParam('limits/gap', gap_limit)
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

        # m.setBoolParam('branching/vanillafullstrong/integralcands', True)  # Whether to consider integer LP solutions as branching candidate variables? Yes: True, No: False
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
    query_expert_prob : float in [0, 1]
                       list, float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
                list, float in [0, 1]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    out_dir2: str
    logfile: str
    iter_index: int
    fullstrong_num: int
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
    query_expert_prob : float in [0, 1]
                       list, float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]
                list, float in [0, 1]
        Maximum running time for an episode, in seconds.
    out_dir2: str
    logfile: str
    iter_index: int
    fullstrong_num: int
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
                    print_step = max(1, n_samples // 100)  ## When the total sample size is greater than 100, print once every 100 samples; when the total sample size is less than 100, print once for each sample.
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
        choices=['case118', 'case300', 'case1888', 'case2383', 'case2848', '24GX', 'case2869',],
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

    base_dir = r"D:\LiJiamigFile\CAMBranch-ljmdata\data\instances\case2383-CSG2024.12.1-v2"
    finished_dir = r"D:/LiJiamigFile/CAMBranch-ljmdata/KIDA_data/samples/case2383_finished_milp.csv"
    out_dir = f'D:/LiJiamigFile/CAMBranch-ljmdata/KIDA_data/samples/{args.problem}_iter{args.iter}'

    train_milp_begin = 1
    train_milp_end = 2001
    valid_milp_begin = 2001
    valid_milp_end = 2401
    test_milp_begin = 2401
    test_milp_end = 2601

    # create output directory, throws an error if it already exists
    os.makedirs(out_dir)
    logfile = os.path.join(out_dir,f'{args.problem}-collect-samples-log.txt')
    utilities_v2.log(f"MILP instance dir: {base_dir}", logfile)

    instances_train = []
    instances_valid = []
    instances_test = []
    for i in range(train_milp_begin, train_milp_end):
        folder_name = f"train_milp/{args.problem}_{i}"
        file_path = os.path.join(base_dir, folder_name, f"{args.problem}_{i}.lp")
        done_file_list = utilities_pinfo.load_done_milp_paths(os.path.join(base_dir, 'train_milp'), finished_dir)
        if os.path.isfile(file_path) and f'{args.problem}_{i}' not in done_file_list:
            instances_train.append(file_path)
    for i in range(valid_milp_begin, valid_milp_end):
        folder_name = f"valid_milp/{args.problem}_{i}"
        file_path = os.path.join(base_dir, folder_name, f"{args.problem}_{i}.lp")
        done_file_list = utilities_pinfo.load_done_milp_paths(os.path.join(base_dir, 'valid_milp'), finished_dir)
        if os.path.isfile(file_path) and f'{args.problem}_{i}' not in done_file_list:
            instances_valid.append(file_path)
    for i in range(test_milp_begin, test_milp_end):
        folder_name = f"test_milp/{args.problem}_{i}"
        file_path = os.path.join(base_dir, folder_name, f"{args.problem}_{i}.lp")
        done_file_list = utilities_pinfo.load_done_milp_paths(os.path.join(base_dir, 'test_milp'), finished_dir)
        if os.path.isfile(file_path) and f'{args.problem}_{i}' not in done_file_list:
            instances_test.append(file_path)

    utilities_v2.log(f"{len(instances_train)} train instances for {train_size} samples", logfile)
    utilities_v2.log(f"{len(instances_valid)} validation instances for {valid_size} samples", logfile)
    utilities_v2.log(f"time_limit[CAMbranch, FSB]: {time_limit}", logfile)                              #

    rng = np.random.RandomState(args.seed)
    collect_samples(instances_train, out_dir + '/train', rng, train_size,
                    args.njobs, exploration_policy=exploration_strategy,
                    query_expert_prob=node_record_prob,
                    time_limit=time_limit, out_dir2=out_dir, logfile=logfile,
                    iter_index=args.iter, fullstrong_num=rng.randint(args.fullstrong_num_min, args.fullstrong_num_max)
                    )


    rng = np.random.RandomState(args.seed + 1)
    collect_samples(instances_valid, out_dir + '/valid', rng, valid_size,
                    args.njobs, exploration_policy=exploration_strategy,
                    query_expert_prob=node_record_prob,
                    time_limit=time_limit, out_dir2=out_dir, logfile=logfile,
                    iter_index=args.iter, fullstrong_num=rng.randint(args.fullstrong_num_min, args.fullstrong_num_max)
                    )

    # rng = np.random.RandomState(args.seed + 2)
    # collect_samples(instances_test, out_dir + '/test', rng, test_size,
    #                 args.njobs, exploration_policy=exploration_strategy,
    #                 query_expert_prob=node_record_prob,
    #                 time_limit = time_limit, out_dir2=out_dir, logfile=logfile,
    #                 iter_index = args.iter, fullstrong_num = args.fullstrong_num
    #                 )
