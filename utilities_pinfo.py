import math
import os.path
import sys
import pandas as pd
import numpy as np
import re
from typing import Dict, Any
from pyscipopt import Model
import openpyxl
import os
import csv

class HUnitparam:#火电参数类
    pmax = 0  # 最大功率(MW)
    pmin = 0  # 最小功率(MW)
    busid = 0  # 所在母线
    fenduan_num = 0  # 分段数
    lowprice = 0  # 最低发电价格(元）
    fenduan_left = []  # 分段区间的左端点（MW)
    fenduan_right = []  # 分段区间的右端点（MW）
    fenduan_V = []  # 分度区间的价格（元/MW）
    iniP = 0  # 初始功率(MW)
    iniState = 0 # 初始运行状态,0：停机,1：运行
    iniT = 0  # 初始状态已运行时间(15分钟)
    RU = 0  # 爬坡（MW/15分钟）
    RD = 0  # 滑坡
    SU = 0  # 开机爬坡
    SD = 0  # 关机滑坡
    minontime = 0  # 最小运行时间(15分钟)
    minofftime = 0  # 最小停机时间(15分钟)
    hotstartcost = 0  # 热启动费用（元）
    coldstartcost = 0  # 冷启动费用（元）
    coldstarttime = 0  # 冷启动时间(15分钟)
    # zone = 0  # 机组分省区域
    def G(self):#初始时刻后仍需连续运行时间
        if(self.iniP>0):
            G=self.minontime-self.iniT
            if(G<0):
                return 0
            else:
                return G
        else:
            return 0
    def L(self):#初始时刻后仍需连续运行时间
        if(self.iniP==0):
            L=self.minofftime+self.iniT
            if(L<0):
                return 0
            else:
                return L
        else:
            return 0

    def __init__(self,pmin,pmax):
        self.pmin=pmin
        self.pmax=pmax
        self.busid=0
        self.fenduan_num = 0
        self.lowprice = 0
        self.fenduan_left = []
        self.fenduan_right = []
        self.fenduan_V = []
        self.avgcost = 0  # 平均分段报价费用
        # self.hotstartcost = 0  # 热启动费用   原始已有
        # self.coldstartcost = 0  # 冷启动费用  原始已有
        self.iniT = 0  # 初始运行时段归一化
        self.t = [calscoreformu1(i) for i in range(1, 25)]  # 时段价值
        # self.t = [calscoreformu1(i) for i in range(1, 97)]  # 请根据总时段数调整,
        self.keepOffT = 0   # ini起仍在最小停机时间内的时段数
        self.keepOnT = 0    # ini起仍在最小开机时间内的时段数
        self.avgcost_scale = 0        # 平均分段报价费用归一化
        self.hotstartcost_scale = 0   # 热启动费用归一化
        self.coldstartcost_scale = 0  # 冷启动费用归一化
        self.iniT_scale = 0           # 初始运行时段归一化
        self.t_scale = [calscoreformu1(i/12.5) for i in range(1, 25)]    # 时段价值归一化
        # self.t_scale = [calscoreformu1(i/48.5) for i in range(1, 97)]  # 请根据总时段数调整,其中,12.5表示range(1, 25)的平均数,48.5表示range(1, 97)的平均数
        self.keepT_scale = 0          # 初始仍需保持状态时段数归一化
        self.beyondT_scale = 0         # 初始超越最小连续状态时段数归一化

    def addfenduan(self,l,r,v):
        self.fenduan_num+=1
        self.fenduan_left.append(l)
        self.fenduan_right.append(r)
        self.fenduan_V.append(v)

    def calavgcost(self): # 计算机组的平均分段报价成本
    # 参考文献: <Trans. on Power System>2025.1 李佩杰、廖畅涛等:
    # <User-Induced Heuristics for Security-Constrained Unit Commitment: Variable Influence Diving and Variable Significance Neighborhood Search>第6页式(33)
        tmp1 = [ self.fenduan_V[m] * (self.fenduan_right[m]-self.fenduan_left[m]) for m in range(0, self.fenduan_num) ]
        # tmp2 = [ (self.fenduan_right[m]-self.fenduan_left[m]) for m in range(0, self.fenduan_num) ]
        # self.avgcost=(self.lowprice+sum(tmp1)) / (self.pmin+sum(tmp2))
        self.avgcost = (self.lowprice + sum(tmp1)) / self.pmax


def getunitdata(unitdataname):# 火电方法
    unitlist = []
    fp = open(unitdataname, "r")
    unitcontext = fp.readlines()
    fp.close()

    for index in unitcontext:
        list = index.split(',')
        if (list[0].isnumeric()):  #  从第二行开始获取数据  list0_trip = list[0].strip('\n')  # 移除字符串末尾的换行符? 正式使用前请检查csv文件的list[0]末尾字符是否为数字
            pmax = float(list[2])
            pmin = float(list[3])
            Hunit = HUnitparam(pmin, pmax)  #  火电类实例化
            Hunit.busid = int(list[1])
            Hunit.iniP = float(list[4])
            if Hunit.iniP > 0.0:
                Hunit.iniState = 1
            Hunit.iniT = int(list[5])
            Hunit.minontime = int(list[6])
            Hunit.minofftime = int(list[7])
            Hunit.coldstarttime = int(list[8])
            if Hunit.iniState == 0:
                # Hunit.keepT = (Hunit.minofftime - abs(Hunit.iniT)) if (Hunit.minofftime - abs(Hunit.iniT) > 0) else 0
                # Hunit.beyondT = (abs(Hunit.iniT) - Hunit.minofftime) if (abs(Hunit.iniT) - Hunit.minofftime > 0) else 0
                Hunit.keepOffT = (Hunit.minofftime - abs(Hunit.iniT) > 0) if (Hunit.minofftime - abs(Hunit.iniT) > 0) else 0  # 仍在最小停机时间内的时段数
            else:
                Hunit.keepOnT  = (Hunit.minontime - abs(Hunit.iniT) > 0) if (Hunit.minontime - abs(Hunit.iniT) > 0) else 0    # 仍在最小开机时间内的时段数
            Hunit.RU = float(list[9])
            Hunit.RD = float(list[10])
            # Hunit.startup_times = int(list[11])
            Hunit.hotstartcost = float(list[12])
            Hunit.coldstartcost = float(list[13])
            lowprice = float(list[14])
            fenduanshu = int(list[15])
            # Hunit.zone = int(list[37])
            Hunit.SU = float(list[38])
            Hunit.SD = float(list[39])
            Hunit.lowprice = lowprice
            if (fenduanshu > 0):
                for m in range(0, fenduanshu):
                    Hunit.addfenduan(float(list[16 + m]), float(list[17 + m]), float(list[27 + m]))  # 这个地方太抽象了,指的是把HUnitparam类中与分段报价相关的属性送给unitlist
            # Hunit.calavgcost()  # 当Hunit实例的分段报价数据更新后,我们接着更新 机组的平均分段报价成本
            unitlist.append(Hunit)

    return unitlist


def getPlineTrueIndex(linelist, Blocklinecsvpath):
    # 通过线路参数列表linelist 构建 (i, j) -> 真实线路索引 的映射字典
    line_dict = {(line.Ni, line.Nj): idx for idx, line in enumerate(linelist)}  # 备注: 此处的.Ni, .Nj要与IEEE_g.py的Lineparam类匹配,否则就会报错没有这个属性!

    PlineTrueIndex = []
    fp = open(Blocklinecsvpath, "r")
    linecontext = fp.readlines()
    fp.close()

    for index in linecontext:
        list = index.split(',')
        if (list[0].isnumeric()):  #  从第二行开始获取数据  list0_trip = list[0].strip('\n')  # 移除字符串末尾的换行符? 正式使用前请检查csv文件的list[0]末尾字符是否为数字
            i, j = int(list[1]), int(list[2])
            ture_index = line_dict.get((i, j), -1)  # 如果在linelist中找不到某条阻塞线路,就赋值-1
            if ture_index == -1:
                print(f"Warning: Line (i={i}, j={j}) not found in linelist")
            PlineTrueIndex.append(ture_index)

    return PlineTrueIndex


def calscoreformu1(x):
    try:
        x_float = float(x)
        return math.exp(-x_float) / (1 + math.exp(-x_float))
    except (TypeError, ValueError) as e:
        print(f"{x} 无法转化为浮点数 {e}...")
        return x
#   适用场景: x需要归一化到（0,0.5],且 x越大,归一化值越小;随着x增大,归一化值下降速度越来越慢
#   归一化公式1:
#   e^(-x) / [1+e^(-x)]

def calscoreformu2(x):
    try:
        x_float = float(x)
        return x_float / (1 + x_float)
    except (TypeError, ValueError) as e:
        print(f"{x} 无法转化为浮点数 {e}...")
        return x
#   适用场景: x需要归一化到[0,1),且 x越大,归一化值越大;随着x增大,归一化值上升速度越来越慢
#   归一化公式2:
#   x / [1+x]

# matchVarId类和parse_candidate_vars函数 用于提取bin类型的候选变量的身份信息,后续进行物理信息评分计算
class matchVarId:                                       # 用于获取指定的候选变量的身份属性,例如 t_uit(1_2)表示 变量uit的name是uit,genid是1,t是2
    def __init__(self, name, genid, t):
        self.name = name   # uit, yit, zit, ycoldit, ustorec, ustored
        self.genid = genid # 发电机序号
        self.t = t         # 时段序号

def parse_candidate_vars(candidate_vars_name):
    pattern = r'^t_(uit|yit|zit|ycoldit|ustorec|ustored)\((\d+)_(\d+)\)$'
    result = []
    for var in candidate_vars_name:
        match = re.match(pattern, var.strip())
        # 报错内容:AttributeError: 'pyscipopt.scip.Variable' object has no attribute 'strip',由此可知var必须是str类型,不允许是pyscipopt.scip.Variable
        # match = re.match(pattern, var)
        if match:
            x, y, z = match.groups()
            result.append(matchVarId(name=x, genid=int(y), t=int(z)))
        else:
            # raise ValueError(f"无效的候选变量名称格式: {var}")  # 写法1: 期望抛出匹配异常
            result.append(matchVarId(name=0, genid=0, t=0))  # 写法2:  直接给出空值
    return result


# SCIPResultExtractor类用于输出机组组合结果,以便对比不同的分支获得的最终结果是否存在差异
# 调用样例
# # 1. 创建提取器（一次创建，多次使用）
# extractor = SCIPResultExtractor()
# # 2. 求解模型（你的原有代码）
# m = scip.Model()
# m.readProblem(f"{instance['path']}")
# m.optimize()
# # 3. 一键提取 results
# results = extractor.extract(m)
# # 4. 使用结果
# print(results['pit']['1'][0])  # 访问 pit(1_0) 的值
class SCIPResultExtractor:
    """
    用于从 PySCIPOpt 模型中提取特定命名模式的变量解值，生成嵌套字典 results。
    支持自定义变量模式，自动处理解值提取。
    """

    # 默认变量模式（可覆盖）
    DEFAULT_PATTERNS = {
        # 'pit': r'pit\((\d+)_(\d+)\)',
        # 'hydroPG': r'hydroPG\((\d+)_(\d+)\)',
        # 'hIN': r'hIN\((\d+)_(\d+)\)',
        # 'hOUT': r'hOUT\((\d+)_(\d+)\)',
        # 'ABQ': r'ABQ\((\d+)_(\d+)\)',
        # 'CABQ': r'CABQ\((\d+)_(\d+)\)',
        # 'BP': r'BP\((\d+)_(\d+)\)',
        # 'CHE': r'CHE\((\d+)_(\d+)\)',
        # 'ACTieLineP': r'ACTieLineP\((\d+)_(\d+)\)',
        # 'DCTieLineP': r'DCTieLineP\((\d+)_(\d+)\)',
        # 'pDM': r'pDM\((\d+)_(\d+)\)',
        # 'hydroPP': r'hydroPP\((\d+)_(\d+)\)',
        'uit': r'uit\((\d+)_(\d+)\)',
        'pit': r'pit\((\d+)_(\d+)\)',
        'yit': r'yit\((\d+)_(\d+)\)',
        'zit': r'zit\((\d+)_(\d+)\)',
        'ycoldit': r'ycoldit\((\d+)_(\d+)\)',
        'Pline': r'Pline\((\d+)_(\d+)\)',  # 
        'ustorec': r'ustorec\((\d+)_(\d+)\)',  # 匹配 ustorec(序号_时段)
        'ustored': r'ustored\((\d+)_(\d+)\)',  # 匹配 ustored(序号_时段)
        'pstorec': r'pstorec\((\d+)_(\d+)\)',  # 匹配 pstorec(序号_时段)
        'pstored': r'pstored\((\d+)_(\d+)\)',  # 匹配 pstored(序号_时段)
    }

    def __init__(self, patterns: Dict[str, str] = None):
        """
        :param patterns: 自定义变量正则模式字典，若为 None 则使用 DEFAULT_PATTERNS
        """
        self.patterns = patterns or self.DEFAULT_PATTERNS
        # 预编译正则表达式，提升匹配效率
        self._compiled_patterns = {k: re.compile(v) for k, v in self.patterns.items()}
    # 函数头注释        参数名1:类型    参数名2:      类型        函数返回值的结构说明
    def extract(self, model: Model, output_xlsx: str='') -> Dict[str, Dict[str, Dict[int, float]]]:
        """
        从已求解的 SCIP 模型中提取变量值。

        :param model: 已调用 optimize() 的 pyscipopt.Model 实例
        :return: results 字典，结构: {var_type: {unit_id: {time_id: value}}}
        """
        # 检查是否有可行解
        # if model.getNSols() == 0:    # 适用于scip9.2.2的pyscipopt
        if model.getObjVal() is None:  # 适用于scip6.0.1的pyscipopt
            print(f"Problem: {model.getProbName()}. There is no primal solution. Skip......")
            return {var_type: {} for var_type in self._compiled_patterns}

        results = {var_type: {} for var_type in self._compiled_patterns}
        all_vars = model.getVars()

        # 高效遍历：每个变量只匹配一次
        for var in all_vars:
            var_name = var.name
            for var_type, pattern in self._compiled_patterns.items():
                match = pattern.match(var_name)
                if match:
                    unit_id, time_str = match.groups()
                    time_id = int(time_str)
                    unit_id = str(unit_id)  # 保持一致性，键为字符串

                    if unit_id not in results[var_type]:
                        results[var_type][unit_id] = {}
                    results[var_type][unit_id][time_id] = model.getVal(var)
                    break  # 匹配成功后跳出，避免重复匹配

        # return results   # ljm debug
        # 将结果转换为DataFrame，并确保按时段升序排列
        dfs = {}
        for var_type, data in results.items():
            df = pd.DataFrame.from_dict({int(unit_id): pd.Series(data[unit_id]) for unit_id in data},orient='index')
            df = df.sort_index(axis=0).sort_index(axis=1, key=lambda x: x.astype(int))  # 确保按时段升序排序
            df.index.name = 'Unit/Plant ID'
            df.columns.name = 'Time Period'
            dfs[var_type] = df

        # 收集模型求解总览信息
        scip_version = model.version()
        model_name = model.getProbName()
        status = model.getStatus()
        stime = model.getSolvingTime()  # SCIP记录的求解器时间t3, 关系: t1 >=t2, t1>=t3, t2和t3大小关系不确定
        gap_100percent = str(format(100 * model.getGap(), '.2')) + '%'
        model_primal =model.getPrimalbound()  # 模型最优目标值
        model_dual = model.getDualbound()  # 模型的最优对偶下界
        nnodes = model.getNNodes()  # SCIP记录的创建的节点总数,表示从根节点出发，SCIP 创建了多少个子节点（包括被剪掉的、求解过的、未处理的等）
        nlps = model.getNLPs()
        info_dict = {
            "Metric": ["SCIP Version", "Model Name", "Model Status", "Model Runtime","Gap%", "Objective Value", "Dual Bound", 'Total Nodes', 'Total LPs'],
            "Value": [scip_version, model_name, status, stime, gap_100percent, model_primal, model_dual, nnodes, nlps]
        }
        info_df = pd.DataFrame(info_dict)

        # 写入Excel文件
        with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
            info_df.to_excel(writer, sheet_name='Solve Info', index=False)  # 总览信息写入第一个工作表
            for var_type, df in dfs.items():                                # 其他信息写入其他工作表
                df.to_excel(writer, sheet_name=var_type)

        print(f"\tResults saved to {output_xlsx}")
        return results
        # return results   # ljm debug 假如我们需要提前返回字典results,也可以将如上代码注释+++++++++++++++++++++++++++++++++++++

    def update_patterns(self, new_patterns: Dict[str, str]):
        """动态更新匹配模式"""
        self.patterns.update(new_patterns)
        self._compiled_patterns = {k: re.compile(v) for k, v in self.patterns.items()}


# load_done_milp_paths函数用于获取csv_path中存放的文件名称,将他们分别与base_path组合后存入列表done_file_list
# 我们重启collect samples后不再对done_file_list列表内的milp收集分支样本
# 调用样例
# 1、base_path = r"./data/instances/case2869/train_milp"     # milp的位置
# 2、csv_path = r"A\your_file_list.csv"                                                      # 已求解过的milp清单路径
# 3、done_file_list = load_done_file_paths(base_path, csv_path)                              # 获取已求解过的milp的名称的列表
def load_done_milp_paths(base_path, csv_path):
    """
    从指定的 CSV 文件中读取文件名（第一列），并与 base_path 拼接，生成完整路径列表。
    Args:
        base_path (str): 文件名称，例如： case118_1、24GX_750
        csv_path (str): 包含文件名的 CSV 文件路径
    Returns:
        list: 完整文件路径列表
    """
    done_file_list = []
    with open(csv_path, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        # next(reader, None)  # 可选: 跳过第一行（None 是默认值，防止空文件报错）
        for row in reader:
            if row:  # 确保行非空
                filename = row[0].strip()  # 只取第一列，并去除空白字符
                full_path = os.path.join(base_path, filename, f'{filename}.lp')
                if filename and os.path.isfile(full_path):  # 避免空字符串
                    done_file_list.append(filename)
    return done_file_list

