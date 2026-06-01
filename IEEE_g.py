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

def getLinedata(filename, busnum, slack, output_PTDF_path):
    PlinelimitCount = 0
    Linelist = []
    fp = open(filename, "r")
    unitcontext = fp.readlines()
    fp.close()
    for index in unitcontext:
        list = index.split(',')
        if (list[0].isnumeric()):
            if list[7].strip() == 'YES':
                PlinelimitCount += 1
            Linelist.append(Lineparam(list))

    if os.path.isfile(output_PTDF_path) is False:
        linenum = len(Linelist)
        B_line = np.zeros((linenum, busnum))
        B_bus = np.zeros((busnum, busnum))
        for index in range(0, len(Linelist)):

            if Linelist[index].ratio == 0:
                ratio = 1
            else:
                ratio = Linelist[index].ratio
            bij = 1 / (Linelist[index].X * ratio)
            From = Linelist[index].Ni - 1
            To = Linelist[index].Nj - 1
            B_line[index, From] += bij
            B_line[index, To] += -bij

            B_bus[From, From] += bij
            B_bus[To, To] += bij
            B_bus[To, From] += -bij
            B_bus[From, To] += -bij

        ref = 1
        B_bus_temp = np.delete(B_bus, ref - 1, axis=1)
        B_bus_temp = np.delete(B_bus_temp, slack - 1, axis=0)
        B_line_temp = np.delete(B_line, ref - 1, axis=1)
        PTDF_temp = B_line_temp @ np.linalg.inv(B_bus_temp)
        zero_column = np.zeros((linenum, 1))
        if slack == 1:
            PTDF = np.hstack((zero_column, PTDF_temp))
        else:
            PTDF = np.hstack((PTDF_temp[:, :slack - 1], zero_column, PTDF_temp[:, slack - 1:]))

        mask_positive = (PTDF > -1e-5) & (PTDF < 0)
        mask_negative = (PTDF < 1e-5) & (PTDF > 0)
        PTDF[mask_positive | mask_negative] = 0

        column_names = [str(i + 1) for i in range(busnum)]
        row_names = ['线路序号' + str(i + 1) for i in range(linenum)]
        PTDF_df = pd.DataFrame(PTDF, columns=column_names, index=row_names)
        PTDF_df.insert(0, '', row_names)
        PTDF_df.to_csv(output_PTDF_path, index=False, header=True)
        print(f"PTDF has been saved to {output_PTDF_path}!")

    return PlinelimitCount, Linelist

def getPTDFdata(filename):
    with open(filename, "r") as fp:
        unitcontext = fp.readlines()
    headers = unitcontext[0].strip().split(",")[1:]
    data_dict = {header: [] for header in headers}
    for row in unitcontext[1:]:
        row_data = row.strip().split(",")[1:]
        if not row_data or not row_data[0]:
            continue
        for header, value in zip(headers, row_data):
            value = float(value)
            data_dict[header].append(value)
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
    T = 24
    period_length = 60
    # T = 96
    # period_length = 15
    model.Trange = pye.RangeSet(1, T)
    model.line = pye.RangeSet(1, len(linelist))
    model.Pline = pye.Var(model.line, model.Trange, within=pye.Reals)
    model.currentline = pye.Param(initialize=1, mutable=True, within=pye.Integers)
    model.Nrange = pye.RangeSet(1, Hunitnum)
    model.CHit = pye.Var(model.Nrange, model.Trange, within=pye.NonNegativeReals)
    model.pit = pye.Var(model.Nrange, model.Trange, within=pye.NonNegativeReals)
    # model.pitm = pye.Var(model.Nrange, model.Trange, pye.RangeSet(1, 10),
    model.pitm = pye.Var(model.Nrange, model.Trange, pye.RangeSet(1, 4),
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
    # model.pstorecm = pye.Var(model.Srange, model.Trange, pye.RangeSet(1, 10), within=pye.NonNegativeReals)
    model.pstorecm = pye.Var(model.Srange, model.Trange, pye.RangeSet(1, 4), within=pye.NonNegativeReals)
    # model.pstoredm = pye.Var(model.Srange, model.Trange, pye.RangeSet(1, 10), within=pye.NonNegativeReals)
    model.pstoredm = pye.Var(model.Srange, model.Trange, pye.RangeSet(1, 4), within=pye.NonNegativeReals)
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

    #Energy Storage
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

    # model.sys_con3 = pye.Constraint(model.Nrange, model.Trange, pye.RangeSet(1, 10), rule=sys_con3)
    model.sys_con3 = pye.Constraint(model.Nrange, model.Trange, pye.RangeSet(1, 4), rule=sys_con3)

    #Energy Storage
    def sys_store3(model, i, t, m, type):
        if type == 1:
            if (m <= storelist[i - 1].C_num):
                fenduanPmax = storelist[i - 1].C_right[m - 1] - storelist[i - 1].C_left[
                    m - 1]
                return model.pstorecm[i, t, m] <= fenduanPmax
            else:
                return model.pstorecm[i, t, m] == 0
        else:
            if (m <= storelist[i - 1].D_num):
                fenduanPmax = storelist[i - 1].D_right[m - 1] - storelist[i - 1].D_left[m - 1]
                return model.pstoredm[i, t, m] <= fenduanPmax
            else:
                return model.pstoredm[i, t, m] == 0

    # model.sys_store3 = pye.Constraint(model.Srange, model.Trange, pye.RangeSet(1, 10), pye.RangeSet(1, 2), rule=sys_store3)
    model.sys_store3 = pye.Constraint(model.Srange, model.Trange, pye.RangeSet(1, 4), pye.RangeSet(1, 2), rule=sys_store3)

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
        for i in model.Nrange:
            sumH += model.pit[i, t]
        #Energy Storage
        sumS = 0
        for i in model.Srange:
            sumS += -model.pstorec[i, t] + model.pstored[i, t]
        return sumH + sumS == load[t-1].LoadSum
        # return sumH == load[t - 1].LoadSum

    model.PowerBalance = pye.Constraint(model.Trange, rule=PowerBalance_rule)

    def sys_Requcon1(model, t):
        sumH = 0
        for i in model.Nrange:
            sumH += model.uit[i, t] * unitlist[i - 1].pmax
        return sumH >= load[t - 1].LoadSum + load[t - 1].LoadReserve
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
        return model.Pline[lineNo, t] == sum

    # model.CalPline = pye.Constraint(model.line, model.Trange, rule=CalPline_rule)
    # model.CalPline = pye.Constraint(pye.RangeSet(1, 4), model.Trange, rule=CalPline_rule)
    model.CalPline = pye.Constraint(pye.RangeSet(1, 20), model.Trange, rule=CalPline_rule)


    def Plinelimit_rule(model, lineNo, t):
        if linelist[lineNo - 1].IsbuildPlinelimit == 'YES':
            # count = model.currentline()
            # progress = count / (PlinelimitCount * T) * 100
            # formatted_progress = f"{progress:.4f}"
            # print(f"Current progress of security constraints establishment: {formatted_progress}%")
            # model.currentline.set_value(count + 1)
            return (linelist[lineNo - 1].Pmin, model.Pline[lineNo, t],
                    linelist[lineNo - 1].Pmax)
        else:
            return pye.Constraint.Skip

    model.Plinelimit = pye.Constraint(model.line, model.Trange, rule=Plinelimit_rule)

    milp_name = f"{Dir2}.lp"
    milp_path = os.path.join(Dir1, milp_name)
    model.write(milp_path, "lp", None, {"symbolic_solver_labels": True})







