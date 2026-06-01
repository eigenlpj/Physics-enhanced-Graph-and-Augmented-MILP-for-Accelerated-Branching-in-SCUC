import bisect
import os.path
import sys
import pandas as pd
import numpy as np
import pyomo.environ as pye


# region-----------------------------topology class
class Busparam:
    busid = 0
    inputP = 0
    LoadP = 0

    def __init__(self, busid):
        self.busid = busid
        self.LoadP = 0

class BusHydroNewparam:
    def __init__(self, busid):
        self.busid = busid
        self.inputP = []

class Loadparam:
    t = 0
    LoadSum = 0
    LoadReserve = 0
    LoadRate = 1

    def __init__(self, str):
        self.t = int(str[0])
        self.LoadSum = float(str[1])
        self.LoadReserve = float(str[2])
        self.LoadRate = float(str[3])

class Lineparam:
    Id = 0
    Ni = 0
    Nj = 0
    X = 0
    Pmin = 0
    Pmax = 0
    ratio = 1
    IsbuildPlinelimit = 'NO'

    def __init__(self, str):
        self.Id = int(str[0])
        self.Ni = int(str[1])
        self.Nj = int(str[2])
        self.X = float(str[3])
        self.Pmin = float(str[4])
        self.Pmax = float(str[5])
        self.ratio = float(str[6])
        self.IsbuildPlinelimit = str[7].strip()
# endregion

# region-----------------------------topology method
def getbusdata(busname, busloadname, loadratefile):
    buslist = []
    fp = open(busname, "r")
    buscontext = fp.readlines()
    fp.close()
    for index in buscontext:
        list = index.split(',')
        if (list[0].isnumeric()):
            buslist.append(Busparam(int(list[0])))

    fp = open(busloadname, "r")
    businputcontext = fp.readlines()
    fp.close()
    for index in businputcontext:
        list = index.split(',')
        if (list[0].isnumeric()):
            buslist[int(list[0]) - 1].LoadP = float(list[1])

    fp = open(loadratefile, "r")
    unitcontext = fp.readlines()
    fp.close()
    load = []
    for index in unitcontext:
        list = index.split(',')
        if (list[0].isnumeric()):
            load.append(Loadparam(list))

    return buslist, load

def getbusHydroNewdata(busname, busHydroNew, T):
    buslist = []
    fp = open(busname, "r")
    buscontext = fp.readlines()
    fp.close()
    for index in buscontext:
        list = index.split(',')
        if (list[0].isnumeric()):
            buslist.append(BusHydroNewparam(int(list[0])))

    fp = open(busHydroNew, "r")
    businputcontext = fp.readlines()
    fp.close()
    for index in businputcontext:
        list = index.split(',')
        if (list[0].isnumeric()):
            assert T==len(list)-2, "用户提供的总时段数 <> 水新数据中的总时段数,请检查!"
            for t in range(0, len(list) - 2):
                if (len(buslist[int(list[0]) - 1].inputP) <= t):
                    buslist[int(list[0]) - 1].inputP.append(float(list[2 + t]))
                else:
                    buslist[int(list[0]) - 1].inputP[t] += float(list[2 + t])
    for bi in buslist:
        if (len(bi.inputP)==0):
            for t in range(0, T):
                bi.inputP.append(0)

    return buslist


# 线路首末节点、潮流正反向传输极限、生成功率分布转移因子矩阵文件
def getLinedata(filename, busnum, slack, output_PTDF_path):
    PlinelimitCount = 0                        # 统计应建立安全约束的线路数量
    Linelist = []
    fp = open(filename, "r")
    unitcontext = fp.readlines()
    fp.close()
    for index in unitcontext:
        list = index.split(',')
        if (list[0].isnumeric()):              # 从第二行开始获取数据
            if list[7].strip() == 'YES':
                PlinelimitCount += 1           # 安全约束线路数量+1
            Linelist.append(Lineparam(list))

    if os.path.isfile(output_PTDF_path) is False:  # 检查是否已经有PTDF文件
        linenum = len(Linelist)                    # 获取线路总条数
        B_line = np.zeros((linenum, busnum))       # 初始化B_line为全零矩阵
        B_bus = np.zeros((busnum, busnum))         # 初始化B_bus 为全零矩阵
        for index in range(0, len(Linelist)):

            if Linelist[index].ratio == 0:         # 调整线路的标幺变比
                ratio = 1
            else:
                ratio = Linelist[index].ratio
            bij = 1 / (Linelist[index].X * ratio)  # 线路ij的电纳值
            From = Linelist[index].Ni - 1          # 线路ij的首段索引
            To = Linelist[index].Nj - 1            # 线路ij的末端索引
            # 更新线路电纳矩阵  或者 线路关联矩阵
            B_line[index, From] += bij         # 首端节点对应的元素为正数。注：减一是因为数组第一个元素从0开始
            B_line[index, To] += -bij          # 末端节点对应的元素为负数。注：减一是因为数组第一个元素从0开始

            # 更新节点电纳矩阵
            B_bus[From, From] += bij
            B_bus[To, To] += bij
            B_bus[To, From] += -bij
            B_bus[From, To] += -bij

        # 计算功率分布转移因子矩阵PTDF 并保存为文件
        ref = 1  # 相位参考节点序号,该节点相位为0
        # ref = slack  # 相位参考节点序号,该节点相位为0
        B_bus_temp = np.delete(B_bus, ref - 1, axis=1)         # 删除B_bus的第ref列
        B_bus_temp = np.delete(B_bus_temp, slack - 1, axis=0)  # 删除B_bus的第slack行得到B_bus_temp
        B_line_temp = np.delete(B_line, ref - 1, axis=1)       # 删除B_line的第ref列得到B_line_temp
        PTDF_temp = B_line_temp @ np.linalg.inv(B_bus_temp)
        zero_column = np.zeros((linenum, 1))                   # 需要插入PTDF的第slack列的全零列
        if slack == 1:
            PTDF = np.hstack((zero_column, PTDF_temp))
        else:
            PTDF = np.hstack((PTDF_temp[:, :slack - 1], zero_column, PTDF_temp[:, slack - 1:]))

        # 将PTDF中靠近零的元素替换为零,避免矩阵系数范围太大
        mask_positive = (PTDF > -1e-5) & (PTDF < 0)
        mask_negative = (PTDF < 1e-5) & (PTDF > 0)
        PTDF[mask_positive | mask_negative] = 0

        column_names = [str(i + 1) for i in range(busnum)]                   # 标注矩阵的行号、列号。注意：有定义可知，PTDF的行数是线路总条数，PTDF的列数是节点总个数
        row_names = ['线路序号' + str(i + 1) for i in range(linenum)]
        PTDF_df = pd.DataFrame(PTDF, columns=column_names, index=row_names)  # 将矩阵PTDF转换成DataFrame
        PTDF_df.insert(0, '', row_names)
        PTDF_df.to_csv(output_PTDF_path, index=False, header=True)           # 将数据表转化为csv文件并命名好
        print(f"PTDF has been saved to {output_PTDF_path}!")
    # 2025.8.12 佳明debug+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #     B_bus_2 = np.linalg.inv(B_bus)
    #     Bbus_df = pd.DataFrame(B_bus_2)  # 将矩阵B_bus_逆 转换成DataFrame
    #     Bbus_df.to_csv(output_PTDF_path+'-B_bus.csv', index=False, header=True)  # 将数据表转化为csv文件并命名好
    #     print(f"Bbus逆 has been saved to {output_PTDF_path}!")
    # 2025.8.12 佳明debug+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    return PlinelimitCount, Linelist

# 线路对节点的功率分布转移因子矩阵
def getPTDFdata(filename):
    # 打开文件并读取所有行
    with open(filename, "r") as fp:
        unitcontext = fp.readlines()
    # 假设第一行是列头，去除行尾的换行符并按分隔符（如逗号）分割
    headers = unitcontext[0].strip().split(",")[1:]  # 仅保留从第二列开始的数据,并将他们按逗号分隔开来
    # 初始化一个字典，键为列头，值为空列表
    data_dict = {header: [] for header in headers}
    # 处理每一行数据（从第二行开始）
    for row in unitcontext[1:]:
        row_data = row.strip().split(",")[1:]  # 仅保留从第二列开始的数据,并将他们按逗号分隔开来
        if not row_data or not row_data[0]:    # 如果这一行是空的或者第一项为空，则该行跳过,不读取
            continue
        # 遍历每一列的值，并将值添加到对应的列表中
        for header, value in zip(headers, row_data):
            value = float(value)
            data_dict[header].append(value)   # 字典的键：节点序号，值：线路对该节点的功率分布转移因子组成的列表
    return data_dict
# endregion

# region-----------------------------thermal unit class
class HUnitparam:
    pmax = 0
    pmin = 0
    busid = 0
    fenduan_num = 0
    lowprice = 0
    fenduan_left = []
    fenduan_right = []
    fenduan_V = []
    iniP = 0
    iniState = 0
    iniT = 0
    RU = 0
    RD = 0
    SU = 0
    SD = 0
    minontime = 0
    minofftime = 0
    hotstartcost = 0
    coldstartcost = 0
    coldstarttime = 0
    # zone = 0

    def G(self):
        if (self.iniP > 0):
            G = self.minontime - self.iniT
            if (G < 0):
                return 0
            else:
                return G
        else:
            return 0

    def L(self):
        if (self.iniP == 0):
            L = self.minofftime + self.iniT
            if (L < 0):
                return 0
            else:
                return L
        else:
            return 0

    def __init__(self, pmin, pmax):
        self.pmin = pmin
        self.pmax = pmax
        self.fenduan_num = 0
        self.lowprice = 0
        self.fenduan_left = []
        self.fenduan_right = []
        self.fenduan_V = []

    def addfenduan(self, l, r, v):
        self.fenduan_num += 1
        self.fenduan_left.append(l)
        self.fenduan_right.append(r)
        self.fenduan_V.append(v)
# endregion

# region-----------------------------thermal unit method
def getunitdata(unitdataname):
    unitlist = []
    fp = open(unitdataname, "r")
    unitcontext = fp.readlines()
    fp.close()

    for index in unitcontext:
        list = index.split(',')
        if (list[0].isnumeric()):
            pmax = float(list[2])
            pmin = float(list[3])
            Hunit = HUnitparam(pmin, pmax)
            Hunit.busid = int(list[1])
            Hunit.iniP = float(list[4])
            if Hunit.iniP > 0.0:
                Hunit.iniState = 1
            Hunit.iniT = int(list[5])
            Hunit.minontime = int(list[6])
            Hunit.minofftime = int(list[7])
            Hunit.coldstarttime = int(list[8])
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
                    Hunit.addfenduan(float(list[16 + m]), float(list[17 + m]), float(list[27 + m]))
            unitlist.append(Hunit)
    return unitlist
# endregion

# region-----------------------------Energy storage class
class storeparam:
    pmax = 0
    busid = 0
    Clife = 1
    Qmin = 0
    Qmax = 0
    Q0 = 0
    etaC = 1
    etaD = 1
    C_num = 0
    C_left = []
    C_right = []
    C_cost = []
    D_num = 0
    D_left = []
    D_right = []
    D_cost = []

    def __init__(self, pmax):
        self.pmax = pmax
        self.C_num = 0
        self.C_left = []
        self.C_right = []
        self.C_cost = []
        self.D_num = 0
        self.D_left = []
        self.D_right = []
        self.D_cost = []

    def addC(self, l, r, v):
        self.C_num  += 1
        self.C_left.append(l)
        self.C_right.append(r)
        self.C_cost.append(v)

    def addD(self, l, r, v):
        self.D_num  += 1
        self.D_left.append(l)
        self.D_right.append(r)
        self.D_cost.append(v)
# endregion

# region-----------------------------Energy storage method
def getstoredata(storedataname):
    storelist = []
    fp = open(storedataname, "r")
    storecontext = fp.readlines()
    fp.close()

    for index in storecontext:
        list = index.split(',')
        if (list[0].isnumeric()):
            pmax = float(list[3])
            store = storeparam(pmax)
            store.busid = int(list[2])
            store.Clife = float(list[4])
            store.Qmin = float(list[5])
            store.Qmax = float(list[6])
            store.Q0 = float(list[7])
            store.etaC = float(list[8])
            store.etaD = float(list[9])
            C_segments = int(list[10])
            D_segments = int(list[32])
            if (C_segments > 0):
                for m in range(0, C_segments):
                    store.addC(float(list[11 + m]), float(list[12 + m]), float(list[22 + m]))
            if (D_segments > 0):
                for m in range(0, D_segments):
                    store.addD(float(list[33 + m]), float(list[34 + m]), float(list[44 + m]))
            storelist.append(store)
    return storelist
# endregion


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Please provide the folder path, the base name of the example (case118, case300, case2383...), and the slack node number!")
        sys.exit(1)
    elif len(sys.argv) < 3:
        print("Please provide the base names of the examples (case118, case300, case2383...) and the slack node numbers!")
        sys.exit(1)
    elif len(sys.argv) < 4:
        print("Please provide the slack node number!")
        sys.exit(1)

    Dir1 = sys.argv[1]
    Dir2 = os.path.basename(Dir1)
    caseName = sys.argv[2]
    slackBus = sys.argv[3]

    buslist, load = getbusdata(Dir1 + "\\1-" + caseName + "-母线名称.csv",
                               Dir1 + "\\2-" + caseName + "-母线负荷.csv",
                               Dir1 + "\\3-" + caseName + "-系统负荷.csv")
    PlinelimitCount, linelist = getLinedata(Dir1 + "\\4-" + caseName + "-线路参数.csv", len(buslist), int(slackBus), Dir1 + "\\PTDF_matrix.csv")
    PTDF_dict = getPTDFdata(Dir1 + "\\PTDF_matrix.csv")
    unitlist = getunitdata(Dir1 + "\\5-" + caseName + "-机组数据.csv")
    storelist = getstoredata(Dir1 + "\\6-" + caseName + "-储能电站.csv")

    Hunitnum = len(unitlist)
    storenum = len(storelist)
    Punitlist = [-1] * (len(buslist))
    Pstorelist = [-1] * (len(buslist))
    for index in range(0, len(unitlist)):
        if Punitlist[unitlist[index].busid - 1] == -1:
            Punitlist[unitlist[index].busid - 1] = [index + 1]
        else:
            Punitlist[unitlist[index].busid - 1].append(index + 1)
    for index in range(0, len(storelist)):
        if Pstorelist[storelist[index].busid - 1] == -1:
            Pstorelist[storelist[index].busid - 1] = [index + 1]
        else:
            Pstorelist[storelist[index].busid - 1].append(index + 1)


    model = pye.ConcreteModel()
    # T = 24
    # period_length = 60
    T = 96
    period_length = 15
    model.Trange = pye.RangeSet(1, T)

    busHydroNewlist = getbusHydroNewdata(Dir1 + "\\1-" + caseName + "-母线名称.csv",
                                         Dir1 + "\\3-" + caseName + "-水新.csv",T)

    model.line = pye.RangeSet(1, len(linelist))
    model.Pline = pye.Var(model.line, model.Trange, within=pye.Reals)
    model.currentline = pye.Param(initialize=1, mutable=True, within=pye.Integers)
    model.Nrange = pye.RangeSet(1, Hunitnum)
    model.CHit = pye.Var(model.Nrange, model.Trange, within=pye.NonNegativeReals)
    model.pit = pye.Var(model.Nrange, model.Trange, within=pye.NonNegativeReals)
    model.pitm = pye.Var(model.Nrange, model.Trange, pye.RangeSet(1, 10),
                         within=pye.NonNegativeReals)
    model.uit = pye.Var(model.Nrange, model.Trange, within=pye.Binary)
    model.yit = pye.Var(model.Nrange, model.Trange, within=pye.Binary)
    model.zit = pye.Var(model.Nrange, model.Trange, within=pye.Binary)
    model.ycoldit = pye.Var(model.Nrange, model.Trange, within=pye.Binary)

    # Energy Storage
    model.Srange = pye.RangeSet(1, storenum)
    model.Cstore = pye.Var(model.Srange, model.Trange, within=pye.Reals)
    model.pstorec = pye.Var(model.Srange, model.Trange, within=pye.NonNegativeReals)
    model.pstored = pye.Var(model.Srange, model.Trange, within=pye.NonNegativeReals)
    model.pstorecm = pye.Var(model.Srange, model.Trange, pye.RangeSet(1, 10), within=pye.NonNegativeReals)
    model.pstoredm = pye.Var(model.Srange, model.Trange, pye.RangeSet(1, 10), within=pye.NonNegativeReals)
    model.ustorec = pye.Var(model.Srange, model.Trange, within=pye.Binary)
    model.ustored = pye.Var(model.Srange, model.Trange, within=pye.Binary)
    model.pstoreEB = pye.Var(model.Srange, model.Trange, within=pye.NonNegativeReals)

    def sys_obj(model):
        sum = 0
        for i in model.Nrange:
            for t in model.Trange:
                sum += model.CHit[i, t] + model.yit[i, t] * unitlist[i - 1].hotstartcost + model.ycoldit[i, t] * (
                        unitlist[i - 1].coldstartcost - unitlist[i - 1].hotstartcost)

        #Energy Storage
        for i in model.Srange:
            for t in model.Trange:
                sum += model.Cstore[i, t]
        return sum

    model.SYS_OBJ = pye.Objective(rule=sys_obj, sense=pye.minimize)


    def sys_con1(model, i, t):
        sum = model.uit[i, t] * unitlist[i - 1].lowprice
        for m in range(0, unitlist[i - 1].fenduan_num):
            sum += model.pitm[i, t, m + 1] * unitlist[i - 1].fenduan_V[m]
        return model.CHit[i, t] == sum  * (period_length / 60)

    model.sys_con1 = pye.Constraint(model.Nrange, model.Trange, rule=sys_con1)

    def sys_store1(model, i, t):
        sum = 0
        for m in range(0, storelist[i - 1].C_num):
            sum += -model.pstorecm[i, t, m + 1] * storelist[i - 1].C_cost[m]
        for m in range(0, storelist[i - 1].D_num):
            sum += model.pstoredm[i, t, m + 1] * storelist[i - 1].D_cost[m]
        return model.Cstore[i, t] == sum * (period_length / 60)

    model.sys_store1 = pye.Constraint(model.Srange, model.Trange, rule=sys_store1)

    def sys_con2(model, i, t):
        sum = model.uit[i, t] * unitlist[i - 1].pmin
        for m in range(0, unitlist[i - 1].fenduan_num):
            sum += model.pitm[i, t, m+1]
        return model.pit[i, t] == sum

    model.sys_con2 = pye.Constraint(model.Nrange, model.Trange, rule=sys_con2)

    #Energy Storage
    def sys_store2(model, i, t, type):
        sum = 0
        if type == 1:
            for m in range(0, storelist[i - 1].C_num):
                sum += model.pstorecm[i, t, m + 1]
            return sum == model.pstorec[i, t]
        else:
            for m in range(0, storelist[i - 1].D_num):
                sum += model.pstoredm[i, t, m + 1]
            return sum == model.pstored[i, t]

    model.sys_store2 = pye.Constraint(model.Srange, model.Trange, pye.RangeSet(1, 2), rule=sys_store2)

    def sys_con3(model, i, t, m):
        if (m <= unitlist[i - 1].fenduan_num):
            fenduanPmax = unitlist[i - 1].fenduan_right[m - 1] - unitlist[i - 1].fenduan_left[m - 1]
            return model.pitm[i, t, m] <= fenduanPmax
        else:
            return model.pitm[i, t, m] == 0

    model.sys_con3 = pye.Constraint(model.Nrange, model.Trange, pye.RangeSet(1, 10), rule=sys_con3)

    #Energy Storage
    def sys_store3(model, i, t, m, type):
        if type == 1:
            if (m <= storelist[i - 1].C_num):
                fenduanPmax = storelist[i - 1].C_right[m - 1] - storelist[i - 1].C_left[m - 1]
                return model.pstorecm[i, t, m] <= fenduanPmax
            else:
                return model.pstorecm[i, t, m] == 0
        else:
            if (m <= storelist[i - 1].D_num):
                fenduanPmax = storelist[i - 1].D_right[m - 1] - storelist[i - 1].D_left[m - 1]
                return model.pstoredm[i, t, m] <= fenduanPmax
            else:
                return model.pstoredm[i, t, m] == 0

    model.sys_store3 = pye.Constraint(model.Srange, model.Trange, pye.RangeSet(1, 10), pye.RangeSet(1, 2), rule=sys_store3)

    def sys_con4(model, i, t, type):
        if (type == 1):
            return model.ycoldit[i, t] <= model.yit[i, t]
        else:
            krange = range(unitlist[i-1].minofftime + 1,
                           unitlist[i-1].minofftime + unitlist[i-1].coldstarttime + 2)
            return model.ycoldit[i, t] >= (model.yit[i, t] - sum(model.uit[i, t - k] for k in krange if t - k > 0))

    model.sys_con4 = pye.Constraint(model.Nrange, model.Trange, pye.RangeSet(1, 2), rule=sys_con4)

    def PowerBalance_rule(model, t):
        sumH = 0
        sumS = 0
        for i in model.Nrange:
            sumH += model.pit[i, t]

        #Energy Storage
        for i in model.Srange:
            sumS += -model.pstorec[i, t] + model.pstored[i, t]

        sumL = sum(bi.inputP[t-1] for bi in busHydroNewlist)
        return sumH + sumS == load[t - 1].LoadSum + sumL

    model.PowerBalance = pye.Constraint(model.Trange, rule=PowerBalance_rule)


    def sys_Requcon1(model, t):
        sumH = 0
        for i in model.Nrange:
            sumH += model.uit[i, t] * unitlist[i - 1].pmax

        sumL = sum(bi.inputP[t-1] for bi in busHydroNewlist)
        return sumH >= load[t - 1].LoadSum + load[t - 1].LoadReserve + sumL

    model.sys_Requcon1 = pye.Constraint(model.Trange, rule=sys_Requcon1)

    def UnitPlimit_con(model, i, t, type):
        if type == 1:
            return model.pit[i, t] <= model.uit[i, t] * unitlist[i - 1].pmax
        else:
            return model.pit[i, t] >= model.uit[i, t] * unitlist[i - 1].pmin

    model.UnitPlimit_con = pye.Constraint(model.Nrange, model.Trange, pye.RangeSet(1, 2), rule=UnitPlimit_con)

    def PupLimit_con(model, i, t, type):
        if (t == 1):
            ptjy = unitlist[i - 1].iniP
            if (unitlist[i - 1].iniP > 0):
                utjy = 1
            else:
                utjy = 0
        else:
            ptjy = model.pit[i, t - 1]
            utjy = model.uit[i, t - 1]
        if (type == 1):
            return model.pit[i, t] - ptjy <= utjy * unitlist[i - 1].RU + model.yit[i, t] * unitlist[i - 1].SU
        else:
            return ptjy - model.pit[i, t] <= model.uit[i, t] * unitlist[i - 1].RD + model.zit[i, t] * unitlist[i - 1].SD

    model.PupLimit_con = pye.Constraint(model.Nrange, model.Trange, pye.RangeSet(1, 2), rule=PupLimit_con)

    def logic_con(model, i, t):
        if (t == 1):
            if (unitlist[i - 1].iniP > 0):
                utjy = 1
            else:
                utjy = 0
        else:
            utjy = model.uit[i, t - 1]
        return model.uit[i, t] - utjy == model.yit[i, t] - model.zit[i, t]

    model.logic_con = pye.Constraint(model.Nrange, model.Trange, rule=logic_con)

    def minon_con(model, i, t, type):
        if (type == 1):
            if (t > unitlist[i - 1].G()):
                kmin = t - unitlist[i - 1].minontime + 1
                if (kmin < 1):
                    kmin = 1
                sum = 0
                for k in range(kmin, t + 1):
                    sum += model.yit[i, k]
                return sum <= model.uit[i, t]
            else:
                return model.uit[i, t] == 1
        else:
            if (t > unitlist[i - 1].L()):
                kmin = t - unitlist[i - 1].minofftime + 1
                if (kmin < 1):
                    kmin = 1
                sum = 0
                for k in range(kmin, t + 1):
                    sum += model.zit[i, k]
                return sum <= 1 - model.uit[i, t]
            else:
                return model.uit[i, t] == 0

    model.minon_con = pye.Constraint(model.Nrange, model.Trange, pye.RangeSet(1, 2), rule=minon_con)


    # Energy Storage
    def pstoreLimit_rule(model, i, t, type):
        if type==1:
            return model.pstorec[i, t] <= model.ustorec[i, t] * storelist[i-1].pmax
        else:
            return model.pstored[i, t] <= model.ustored[i, t] * storelist[i-1].pmax

    model.pstoreLimit = pye.Constraint(model.Srange, model.Trange, pye.RangeSet(1, 2), rule=pstoreLimit_rule)

    def pstoreState_rule(model, i, t):
        return model.ustorec[i, t] + model.ustored[i, t] == 1

    model.pstoreState = pye.Constraint(model.Srange, model.Trange, rule=pstoreState_rule)

    def pstoreQ1_rule(model, i, t):
        EB = 0
        if t==1:
            EB = storelist[i-1].Q0
        else:
            EB = model.pstoreEB[i, t-1]
        return model.pstoreEB[i, t] == EB + (storelist[i - 1].etaC * model.pstorec[i, t]
                                             - model.pstored[i, t] / storelist[i - 1].etaD) * (period_length / 60)

    model.pstoreQ1 = pye.Constraint(model.Srange, model.Trange, rule=pstoreQ1_rule)

    def pstoreQ2_rule(model, i, t):
        return (storelist[i - 1].Qmin, model.pstoreEB[i, t], storelist[i - 1].Qmax * storelist[i-1].Clife)

    model.pstoreQ2 = pye.Constraint(model.Srange, model.Trange, rule=pstoreQ2_rule)

    def pstoreQ3_rule(model, i, t):
        return model.pstoreEB[i, t] >= storelist[i - 1].Q0

    model.pstoreQ3 = pye.Constraint(model.Srange, model.Trange, rule=pstoreQ3_rule)


    def CalPline_rule(model, lineNo, t):
        sum = 0
        for busi in range(0, len(buslist)):
            if Punitlist[busi] != -1:
                for i in range(0, len(Punitlist[busi])):
                    sum += model.pit[Punitlist[busi][i], t] * float(PTDF_dict[str(busi + 1)][lineNo - 1])

            # Energy Storage
            if Pstorelist[busi] != -1:
                for i in range(0, len(Pstorelist[busi])):
                    sum += (model.pstored[Pstorelist[busi][i], t] - model.pstorec[Pstorelist[busi][i], t]) * float(
                        PTDF_dict[str(busi + 1)][lineNo - 1])

            sum -= buslist[busi].LoadP * load[t - 1].LoadRate * float(
                PTDF_dict[str(busi + 1)][lineNo - 1])

            # Wind, hydro,
            sum -= busHydroNewlist[busi].inputP[t -1] * float(
                PTDF_dict[str(busi + 1)][lineNo - 1])

        return model.Pline[lineNo, t] == sum

    # model.CalPline = pye.Constraint(model.line, model.Trange, rule=CalPline_rule)
    model.CalPline_A = pye.Constraint(pye.RangeSet(7, 19), model.Trange, rule=CalPline_rule)
    # model.CalPline_B = pye.Constraint(pye.RangeSet(3305, 3308), model.Trange, rule=CalPline_rule)              # 1-2500milp
    model.CalPline_B = pye.Constraint(pye.RangeSet(3304, 3307), model.Trange, rule=CalPline_rule)        # 2500-2600milp


    def Plinelimit_rule(model, lineNo, t):
        if linelist[lineNo - 1].IsbuildPlinelimit == 'YES':
            count = model.currentline()
            progress = count / (PlinelimitCount * T) * 100
            formatted_progress = f"{progress:.4f}"
            # print(f"Current progress of security constraints establishment: {formatted_progress}%")
            model.currentline.set_value(count + 1)
            return (linelist[lineNo - 1].Pmin, model.Pline[lineNo, t],
                    linelist[lineNo - 1].Pmax)
        else:
            return pye.Constraint.Skip

    # model.Plinelimit = pye.Constraint(model.line, model.Trange, rule=Plinelimit_rule)
    model.Plinelimit_A = pye.Constraint(pye.RangeSet(7, 19), model.Trange, rule=Plinelimit_rule)
    # model.Plinelimit_B = pye.Constraint(pye.RangeSet(3305, 3308), model.Trange, rule=Plinelimit_rule)        # 1-2500milp
    model.Plinelimit_B = pye.Constraint(pye.RangeSet(3304, 3307), model.Trange, rule=Plinelimit_rule)  # 2500-2600milp


    milp_name = f"{Dir2}.lp"
    milp_path = os.path.join(Dir1, milp_name)
    model.write(milp_path, "lp", None, {"symbolic_solver_labels": True})







