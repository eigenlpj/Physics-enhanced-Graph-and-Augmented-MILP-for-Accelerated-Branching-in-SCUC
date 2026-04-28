import os.path
import sys
import pandas as pd
import numpy as np
import pyomo.environ as pye


# region-----------------------------网架拓扑类
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

# linedata
class Lineparam:
    Id = 0  # 线路id
    Ni = 0  # 首端节点
    Nj = 0  # 末端节点
    X = 0 # 线路电抗标幺值
    Pmin = 0  # 线路潮流反向传输极限
    Pmax = 0  # 线路潮流正向传输极限
    ratio = 1 # 线路变压器标幺变比：≠0表示拥有变压器;=0表示无变压器,等效标幺变比=1
    IsbuildPlinelimit = 'NO'  # 默认线路不建立安全约束

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

# region-----------------------------网架拓扑方法
# 节点序号、节点负荷、systemload、旋转备用、负荷曲线
def getbusdata(busname, busloadname, loadratefile):
    buslist = []
    fp = open(busname, "r")
    buscontext = fp.readlines()
    fp.close()
    for index in buscontext:
        list = index.split(',')
        if (list[0].isnumeric()):                    #  从第二行开始获取数据
            buslist.append(Busparam(int(list[0])))   #  所有节点序号

    fp = open(busloadname, "r")
    businputcontext = fp.readlines()
    fp.close()
    for index in businputcontext:
        list = index.split(',')
        if (list[0].isnumeric()):
            buslist[int(list[0]) - 1].LoadP = float(list[1])  #  母线固定注入功率(负数)基准值 或 busload预测值(正数)基准值

    fp = open(loadratefile, "r")
    unitcontext = fp.readlines()
    fp.close()
    load = []
    for index in unitcontext:
        list = index.split(',')
        if (list[0].isnumeric()):
            load.append(Loadparam(list))  # 获取各时段的systemload、旋转备用、负荷曲线

    return buslist, load

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

# region-----------------------------火电类
class HUnitparam:
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

    def G(self):  # 初始时刻后仍需连续运行时间
        if (self.iniP > 0):
            G = self.minontime - self.iniT
            if (G < 0):
                return 0
            else:
                return G
        else:
            return 0

    def L(self):  # 初始时刻后仍需连续停机时间
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

# region-----------------------------火电方法
def getunitdata(unitdataname):
    unitlist = []
    fp = open(unitdataname, "r")
    unitcontext = fp.readlines()
    fp.close()

    for index in unitcontext:
        list = index.split(',')
        if (list[0].isnumeric()):  #  从第二行开始获取数据
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
            unitlist.append(Hunit)
    return unitlist
# endregion

# region-----------------------------储能类
class storeparam:
    pmax = 0  # 最大功率(MW)
    busid = 0  # 所在母线
    Clife = 1 # 寿命
    Qmin = 0 # 最小存储电量
    Qmax = 0 # 最大存储电量
    Q0 = 0 # 初始电量
    etaC = 1 # 充电效率
    etaD = 1 # 放电效率
    C_num = 0  # 充电报价分段数
    C_left = []  # 充电分段区间的左端点（MW)
    C_right = []  # 充电分段区间的右端点（MW）
    C_cost = []  # 充电分度区间的价格（元/MW）
    D_num = 0  # 放电报价分段数
    D_left = []  # 放电分段区间的左端点（MW)
    D_right = []  # 放电分段区间的右端点（MW）
    D_cost = []  # 放电分度区间的价格（元/MW）

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

    def addC(self, l, r, v):   # 充电报价分段数、区间左端点、区间右端点、价格
        self.C_num  += 1
        self.C_left.append(l)
        self.C_right.append(r)
        self.C_cost.append(v)

    def addD(self, l, r, v):   # 放电报价分段数、区间左端点、区间右端点、价格
        self.D_num  += 1
        self.D_left.append(l)
        self.D_right.append(r)
        self.D_cost.append(v)
# endregion

# region-----------------------------储能方法
def getstoredata(storedataname):
    storelist = []
    fp = open(storedataname, "r")
    storecontext = fp.readlines()
    fp.close()

    for index in storecontext:
        list = index.split(',')
        if (list[0].isnumeric()):  #  从第二行开始获取数据
            pmax = float(list[3])
            store = storeparam(pmax)   #  储能类实例化
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
                    store.addC(float(list[11 + m]), float(list[12 + m]), float(list[22 + m]))  # 这个地方太抽象了,指的是把storeparam类中与充电分段报价相关的属性送给storelist
            if (D_segments > 0):
                for m in range(0, D_segments):
                    store.addD(float(list[33 + m]), float(list[34 + m]), float(list[44 + m]))  # 这个地方太抽象了,指的是把storeparam类中与放电分段报价相关的属性送给storelist
            storelist.append(store)
    return storelist
# endregion


if __name__ == '__main__':
    # region-----------------------------所有文件存放的文件夹
    if len(sys.argv) < 2:
        print("请提供文件夹路径、算例基础名称(case118、case300、case2383...)、平衡节点号！")
        sys.exit(1)
    elif len(sys.argv) < 3:
        print("请提供算例基础名称(case118、case300、case2383...)、平衡节点号！")
        sys.exit(1)
    elif len(sys.argv) < 4:
        print("请提供平衡节点号！")
        sys.exit(1)

    Dir1 = sys.argv[1]  # 接收文件夹的完整路径
    Dir2 = os.path.basename(Dir1)  # 获取文件夹的名称
    caseName = sys.argv[2]  # 接收算例基础名称
    slackBus = sys.argv[3]  # 接收平衡节点号
    # print(f"当前处理的文件夹: {Dir1}, 算例基础名称: {caseName}, 平衡节点号: {slackBus}")
    # 检查是否提供了足够的命令行参数
    # if len(sys.argv) < 3:  # 至少需要两个参数（脚本名 + 两个路径）
    #     print("请提供两个文件夹路径作为参数！")
    #     print("用法示例：python script.py Dir1 Dir2")
    #     sys.exit(1)
    #
    # # 接收两个路径参数
    # Dir1 = sys.argv[1]  # 第一个文件夹路径
    # Dir2 = sys.argv[2]  # 第二个文件夹路径
    #
    # print(f"原始数据的路径: {Dir1}")
    # print(f"生成文件的路径: {Dir2}")

    # endregion

    # region-----------------------------读节点序号、节点负荷、systemload、旋转备用、负荷曲线
    # example:  buslist, load  = getbusdata(Dir1 + "\\BusNamefile.csv",                    # 读母线序号
    #                                       Dir1 + "\\Busloadfile.csv",                    # 读busload(若系统中所有节点的负荷值都已经标注,则Busload的数据可放入BusName文件;若busload文件中只记录了负荷值>0的节点,不记录负荷值=0的节点信息,会导致节点序号不全，则需要分别读取Busload和BusName文件
    #                                       Dir1 + "\\Systemloadfile")                     # 读systemload、旋转备用、负荷曲线
    # example:  PlinelimitCount, linelist = getLinedata(Dir1 + "linefile", busnum,
    #                                                   slack, output_PTDF_path)           # 应建立安全约束的线路总条数、线路首末节点、线路潮流正反向传输极限、生成功率分布转移因子矩阵文件
    # example:  PTDF_dict = getPTDFdata(Dir1 + "\\PTDFfile.csv")                           # 读功率分布转移因子矩阵
    buslist, load = getbusdata(Dir1 + "\\1-" + caseName + "-busname.csv",
                               Dir1 + "\\2-" + caseName + "-busload.csv",
                               # Dir1 + "\\3-" + caseName + "-systemload+旋转备用+负荷曲线-24时段-算例1.csv")
                               Dir1 + "\\3-" + caseName + "-systemload.csv")
    # buslist, load = getbusdata(Dir1 + "\\1-IEEE118-busname.csv",
    #                      Dir1 + "\\2-IEEE118-busload.csv",
    #                      Dir1 + "\\3-IEEE118-systemload.csv")
    # buslist, load = getbusdata(Dir1 + "\\1-case2383-busname.csv",
    #                            Dir1 + "\\2-case2383-busload.csv",
    #                            Dir1 + "\\3-case2383-systemload+旋转备用+负荷曲线-24时段-算例1.csv")
    PlinelimitCount, linelist = getLinedata(Dir1 + "\\4-" + caseName + "-linedata.csv", len(buslist), int(slackBus), Dir1 + "\\PTDF_matrix.csv")
    # PlinelimitCount, linelist = getLinedata(Dir1 + "\\4-IEEE118-linedata.csv", len(buslist), 69, Dir1 + "\\PTDF_matrix.csv")
    # PlinelimitCount, linelist = getLinedata(Dir1 + "\\4-case2383-linedata.csv", len(buslist), 18,
    #                                         Dir1 + "\\PTDF_matrix.csv")
    # PlinelimitCount, linelist = getLinedata(Dir1 + "\\4-case300-linedata-算例1.csv", len(buslist), 257, Dir1 + "\\PTDF_matrix.csv")
    # PlinelimitCount, linelist = getLinedata(Dir1 + "\\4-case6515rte-linedata.csv", len(buslist), 4714, Dir1 + "\\PTDF_matrix.csv")
    PTDF_dict = getPTDFdata(Dir1 + "\\PTDF_matrix.csv")
    # endregion

    # region-----------------------------读火电数据(注：不考虑启停曲线)
    # example:  unitlist = getunitdata(Dir1 + "\\Thermalunit.csv")
    unitlist = getunitdata(Dir1 + "\\5-" + caseName + "-unitdata.csv")
    # unitlist = getunitdata(Dir1 + "\\5-case2383-unitdata.csv")
    # endregion

    # region-----------------------------读storage数据
    # example:  storelist = getstoredata(Dir1 + "\\Storage.csv")
    storelist = getstoredata(Dir1 + "\\6-" + caseName + "-storage.csv")
    #print("All Data has been read successfully!")
    # endregion

    # region ----------------------------整理数据
    Hunitnum = len(unitlist)               # 火电机组总数
    storenum = len(storelist)              # storage总数
    Punitlist = [-1] * (len(buslist))      # 记录火电机组接入的母线序号。有接入的为母线序号，无接入的为-1。
    Pstorelist = [-1] * (len(buslist))     # 记录storage接入的母线序号。有接入的为母线序号，无接入的为-1。
    for index in range(0, len(unitlist)):  # 寻找该节点的火电功率变量序号
        if Punitlist[unitlist[index].busid - 1] == -1:
            Punitlist[unitlist[index].busid - 1] = [index + 1]        # 装入连接该节点的第一个火电机组
        else:
            Punitlist[unitlist[index].busid - 1].append(index + 1)    # 装入连接该节点的其他火电机组
    for index in range(0, len(storelist)):  # 寻找该节点的储能功率变量序号
        if Pstorelist[storelist[index].busid - 1] == -1:
            Pstorelist[storelist[index].busid - 1] = [index + 1]      # 装入连接该节点的第一个storage
        else:
            Pstorelist[storelist[index].busid - 1].append(index + 1)  # 装入连接该节点的其他storage
    # endregion

    # region-----------------------------建立模型
    model = pye.ConcreteModel()

    # region-----------------------------定义变量、参数
    # 优化时段总数、单个时段的分钟数、时段索引
    T = 24
    period_length = 60
    # T = 96
    # period_length = 15
    model.Trange = pye.RangeSet(1, T)  # 时段索引

    # 线路变量、参数
    model.line = pye.RangeSet(1, len(linelist))  # 线路索引
    model.Pline = pye.Var(model.line, model.Trange, within=pye.Reals)    # 线路功率
    model.currentline = pye.Param(initialize=1, mutable=True, within=pye.Integers)  # 当前建立安全约束的线路序号,设置为可变参数,定义域为实数

    # 火电变量(四类0-1整数变量)、参数
    model.Nrange = pye.RangeSet(1, Hunitnum)  # 火电机组索引
    model.CHit = pye.Var(model.Nrange, model.Trange, within=pye.NonNegativeReals)  # 火电运行成本
    model.pit = pye.Var(model.Nrange, model.Trange, within=pye.NonNegativeReals)  # 火电功率
    model.pitm = pye.Var(model.Nrange, model.Trange, pye.RangeSet(1, 4),
                         within=pye.NonNegativeReals)  # 火电分段报价每一段的功率变量   后续需要改到98时段？
    model.uit = pye.Var(model.Nrange, model.Trange, within=pye.Binary)  # 运行状态0-1变量
    model.yit = pye.Var(model.Nrange, model.Trange, within=pye.Binary)  # 开机操作0-1变量
    model.zit = pye.Var(model.Nrange, model.Trange, within=pye.Binary)  # 停机操作0-1变量
    model.ycoldit = pye.Var(model.Nrange, model.Trange, within=pye.Binary)  # 冷启动0-1变量

    #储储 # 储能变量、参数
    model.Srange = pye.RangeSet(1, storenum)  # storage索引
    model.Cstore = pye.Var(model.Srange, model.Trange, within=pye.Reals)  # v2: 储能运行成本
    model.pstorec = pye.Var(model.Srange, model.Trange, within=pye.NonNegativeReals)  # 充电功率
    model.pstored = pye.Var(model.Srange, model.Trange, within=pye.NonNegativeReals)  # 放电功率
    # model.pstorecm = pye.Var(model.Srange, model.Trange, pye.RangeSet(1, 10), within=pye.NonNegativeReals)  # 分段充电功率  记得 “pye.RangeSet(1, 4),
    model.pstorecm = pye.Var(model.Srange, model.Trange, pye.RangeSet(1, 4), within=pye.NonNegativeReals)
    # model.pstoredm = pye.Var(model.Srange, model.Trange, pye.RangeSet(1, 10), within=pye.NonNegativeReals)  # 分段放电功率  记得 “pye.RangeSet(1, 4),
    model.pstoredm = pye.Var(model.Srange, model.Trange, pye.RangeSet(1, 4), within=pye.NonNegativeReals)
    model.ustorec = pye.Var(model.Srange, model.Trange, within=pye.Binary)  # 充电状态0-1变量
    model.ustored = pye.Var(model.Srange, model.Trange, within=pye.Binary)  # 放电状态0-1变量
    model.pstoreEB = pye.Var(model.Srange, model.Trange, within=pye.NonNegativeReals)  # 存储的电量


    # 目标函数：邓俊师兄论文的式1和式17
    def sys_obj(model):
        sum = 0
        # 火电成本=运行费用+热启动费用+（冷启动费用-热启动费用）
        for i in model.Nrange:
            for t in model.Trange:
                sum += model.CHit[i, t] + model.yit[i, t] * unitlist[i - 1].hotstartcost + model.ycoldit[i, t] * (
                        unitlist[i - 1].coldstartcost - unitlist[i - 1].hotstartcost)
        #储储# # 储能成本=放电成本-充电成本
        for i in model.Srange:
            for t in model.Trange:
                sum += model.Cstore[i, t]
        return sum

    model.SYS_OBJ = pye.Objective(rule=sys_obj, sense=pye.minimize)


    # 火电运行成本：邓俊师兄论文的约束4
    def sys_con1(model, i, t):
        sum = model.uit[i, t] * unitlist[i - 1].lowprice  # 状态变量×最小技术出力价格
        for m in range(0, unitlist[i - 1].fenduan_num):
            sum += model.pitm[i, t, m + 1] * unitlist[i - 1].fenduan_V[m]  # Σ分段出力值×价格
        return model.CHit[i, t] == sum  * (period_length / 60)  # ×每个时段所代表的小时数.机组运行成本:机组状态数是以1小时为1个优化时段的机组状态数的N倍,所以统一到"元/MWh*MW*h"后,需要乘以每个时段所代表的小时数

    model.sys_con1 = pye.Constraint(model.Nrange, model.Trange, rule=sys_con1)  # 添加约束 角标为i, t规则是上述的等式


    #储储 # 储能运行成本：杨老师课题组论文"精准计及大规模储能电池寿命的电力系统经济调度_李中浩"的约束1
    def sys_store1(model, i, t):
        sum = 0
        for m in range(0, storelist[i - 1].C_num):
            sum += -model.pstorecm[i, t, m + 1] * storelist[i - 1].C_cost[m]    # Σ(-分段充电值×充电报价×每个时段所代表的小时数)
        for m in range(0, storelist[i - 1].D_num):
            sum += model.pstoredm[i, t, m + 1] * storelist[i - 1].D_cost[m]    # Σ(分段放电值×放电报价×每个时段所代表的小时数)
        return model.Cstore[i, t] == sum * (period_length / 60)

    model.sys_store1 = pye.Constraint(model.Srange, model.Trange, rule=sys_store1)  # 添加约束 角标为i, t规则是上述的等式


    # 火电分段出力等式：邓俊师兄论文的约束3
    def sys_con2(model, i, t):
        sum = model.uit[i, t] * unitlist[i - 1].pmin
        for m in range(0, unitlist[i - 1].fenduan_num):
            sum += model.pitm[i, t, m+1]
        return model.pit[i, t] == sum

    model.sys_con2 = pye.Constraint(model.Nrange, model.Trange, rule=sys_con2)


    #储储 # 储能分段放电功率、分段充电功率等式：杨老师课题组论文"精准计及大规模储能电池寿命的电力系统经济调度_李中浩"的约束1
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


    # 火电分段区间功率上下限约束：邓俊师兄论文的约束5
    def sys_con3(model, i, t, m):
        if (m <= unitlist[i - 1].fenduan_num):  # 机组的报价分段序号<=机组的报价分段总数
            fenduanPmax = unitlist[i - 1].fenduan_right[m - 1] - unitlist[i - 1].fenduan_left[m - 1]  # 机组在每个分段的出力<=分段区间的功率范围
            return model.pitm[i, t, m] <= fenduanPmax
        else:  # 若机组的报价分段数=0 或 机组的报价分段序号>机组的报价分段总数
            return model.pitm[i, t, m] == 0

    model.sys_con3 = pye.Constraint(model.Nrange, model.Trange, pye.RangeSet(1, 4), rule=sys_con3)

    #储储 # 储能放电功率、分段充电功率上下限约束：杨老师课题组论文"精准计及大规模储能电池寿命的电力系统经济调度_李中浩"的约束1
    def sys_store3(model, i, t, m, type):
        if type == 1:
            if (m <= storelist[i - 1].C_num):  # 机组的报价分段序号<=机组的报价分段总数
                fenduanPmax = storelist[i - 1].C_right[m - 1] - storelist[i - 1].C_left[
                    m - 1]  # 机组在每个分段的出力<=分段区间的功率范围
                return model.pstorecm[i, t, m] <= fenduanPmax
            else:  # 若机组的报价分段数=0 或 机组的报价分段序号>机组的报价分段总数
                return model.pstorecm[i, t, m] == 0
        else:
            if (m <= storelist[i - 1].D_num):  # 机组的报价分段序号<=机组的报价分段总数
                fenduanPmax = storelist[i - 1].D_right[m - 1] - storelist[i - 1].D_left[
                    m - 1]  # 机组在每个分段的出力<=分段区间的功率范围
                return model.pstoredm[i, t, m] <= fenduanPmax
            else:  # 若机组的报价分段数=0 或 机组的报价分段序号>机组的报价分段总数
                return model.pstoredm[i, t, m] == 0

    model.sys_store3 = pye.Constraint(model.Srange, model.Trange, pye.RangeSet(1, 4), pye.RangeSet(1, 2), rule=sys_store3)

    # 冷启动控制变量ycold的约束：邓俊师兄论文的约束19
    # 写法1：卢哥写法
    # def getuit(i, t):
    #     if (t == 0):
    #         if (unitlist[i - 1].iniP > 0):
    #             return 1
    #         else:
    #             return 0
    #     if (t < 0):
    #         if (unitlist[i - 1].iniP > 0):
    #             if (t < 1 - unitlist[i - 1].iniT):
    #                 return 0
    #             else:
    #                 return 1
    #         if (unitlist[i - 1].iniP == 0):
    #             if (t < 1 + unitlist[i - 1].iniT):
    #                 return 1
    #             else:
    #                 return 0
    #     if (t > 0):
    #         return model.uit[i, t]
    # def sys_con4(model, i, t, type):
    #     if (type == 1):
    #         return model.ycoldit[i, t] <= model.yit[i, t]
    #     else:
    #         sum = 0
    #         for k in range(unitlist[i - 1].minofftime + 1,
    #                        unitlist[i - 1].minofftime + unitlist[i - 1].coldstarttime + 1 + 1):
    #             sum += getuit(i, t - k)
    #         return model.ycoldit[i, t] >= model.yit[i, t] - sum
    #
    # model.sys_con4 = pye.Constraint(model.Nrange, model.Trange, pye.RangeSet(1, 2), rule=sys_con4)

    # 写法2：涛哥写法
    def sys_con4(model, i, t, type):
        if (type == 1):
            return model.ycoldit[i, t] <= model.yit[i, t]
        else:
            krange = range(unitlist[i-1].minofftime + 1,
                           unitlist[i-1].minofftime + unitlist[i-1].coldstarttime + 2)
            # 此处注释为涛哥原始写法,暂时不理解用意
            # temp = min(unitlist[i-1].iniT * unitlist[i-1].iniState,
            #        max(unitlist[i-1].minofftime + unitlist[i-1].coldstarttime + 1 - t + 1, 0)) \
            #        + max((unitlist[i-1].minofftime + unitlist[i-1].coldstarttime + 1 - t + 1 +
            #           unitlist[i-1].iniT) * (1 - unitlist[i-1].iniState), 0)
            # return model.ycoldit[i, t] >= (model.yit[i, t] - sum(model.uit[i, t - k] for k in krange if t - k > 0) - temp)
            return model.ycoldit[i, t] >= (model.yit[i, t] - sum(model.uit[i, t - k] for k in krange if t - k > 0))

    model.sys_con4 = pye.Constraint(model.Nrange, model.Trange, pye.RangeSet(1, 2), rule=sys_con4)


    # 系统功率平衡约束：邓俊师兄论文的约束9
    def PowerBalance_rule(model, t):
        sumH = 0
        for i in model.Nrange:  # 火电出力
            sumH += model.pit[i, t]
        #储储
        sumS = 0
        for i in model.Srange:  # storage出力
            sumS += -model.pstorec[i, t] + model.pstored[i, t]   # 充电功率相当于负荷, 放电功率相当于发电机
        return sumH + sumS == load[t-1].LoadSum
        # return sumH == load[t - 1].LoadSum

    model.PowerBalance = pye.Constraint(model.Trange, rule=PowerBalance_rule)


    # 旋转备用约束：邓俊师兄论文的约束10  备注：本模型认为储能不提供备用容量
    def sys_Requcon1(model, t):
        sumH = 0
        for i in model.Nrange:
            sumH += model.uit[i, t] * unitlist[i - 1].pmax
        return sumH >= load[t - 1].LoadSum + load[t - 1].LoadReserve
    model.sys_Requcon1 = pye.Constraint(model.Trange, rule=sys_Requcon1)


    # 机组出力上下限约束：写法1：邓俊师兄论文的约束21、22、23、24、25; 写法2：邓俊师兄论文的约束11

    #写法1：邓俊师兄论文的约束21、22、23
    # # 最小连续运行时间为1个时段(指的是15分钟或者1小时，视loadData数据中的时段总数而定)的机组序号列表Se1,非1的机组序号列表NoSe1
    # Se1 = list()
    # NoSe1 = list()
    # for i in range(0, Hunitnum):
    #     if unitlist[i].minontime == 1:  # 收集最小连续运行时间为1个时段的机组序号
    #         Se1.append(i)
    #     else:
    #         NoSe1.append(i)
    #
    # # calculation(24):Nup,i,t,为满足minontime,当前时段机组至少继续爬坡的时段数
    # N_up = np.zeros((len(unitlist), T))
    # for t in range(1, T+1):
    #     for i in range(0, len(unitlist)):
    #         if unitlist[i].minontime > 2:
    #             s1 = list()
    #             for k in range(1, unitlist[i].minontime):  # range左闭右开
    #                 if (unitlist[i].SU + (k - 1) * unitlist[i].RU <= unitlist[i].pmax) and (t - k >= 0):
    #                     s1.append(k)
    #                 else:
    #                     s1.append(0)
    #             assert len(s1) > 0, (
    #                 f" s1为空,此时i={i},t={t} "
    #             )
    #             N_up[i, t - 1] = max(s1)
    #
    # # calculation(25):Ndown,i,t,为满足minontime,当前时段机组至少继续滑坡的时段数
    # N_down = np.zeros((len(unitlist), T))
    # for t in range(1, T+1):
    #     for i in range(0, len(unitlist)):
    #         if unitlist[i].minontime - int(N_up[i, t - 1]) > 1:
    #             s2 = list()
    #             for k in range(1, unitlist[i].minontime - int(N_up[i - 1, t - 1]) + 1):  # range左闭右开
    #                 if (unitlist[i].SD + (k - 1) * unitlist[i].RD <= unitlist[i].pmax) and (t + k <= T):
    #                     s2.append(k)
    #                 else:
    #                     s2.append(0)
    #             assert len(s2) > 0, (
    #                 f" s2为空,此时i={i},t={t} "
    #             )
    #             N_down[i, t - 1] = max(s2)
    #
    # # constraint(21):机组出力上限约束(21)
    # def output_less1_rule(model, i, t):
    #     if i in NoSe1:
    #         sum1 = sum(
    #             model.yit[i, t + 1 - k] * (
    #                     unitlist[i - 1].pmax - unitlist[i - 1].SU - (k - 1) *
    #                     unitlist[i - 1].RU) for k in
    #             range(1, int(N_up[i - 1, t - 1] + 1)))
    #         sum2 = sum(model.zit[i, t + k] * (
    #                 unitlist[i - 1].pmax - unitlist[i - 1].SD - (k - 1) *
    #                 unitlist[i - 1].RD) for k in
    #                    range(1, int(N_down[i - 1, t - 1] + 1)))
    #         return model.pit[i, t] <= model.uit[i, t] * unitlist[i - 1].pmax - sum1 - sum2
    #     else:
    #         return pye.Expression.Skip  # 暂时不太理解何时约束跳过、何时令约束为pi,t<ui,t * Pmax
    #
    # model.output_less1 = pye.Constraint(model.Nrange, model.Trange, rule=output_less1_rule)
    #
    #
    # # constraint(22):机组出力上限约束(22)
    # def output_less2_rule(model, i, t):
    #     if i in Se1:
    #         if t + 1 in range(0, T):
    #             return model.pit[i, t] <= model.uit[i, t] * unitlist[i - 1].pmax - model.yit[i, t] * (
    #                     unitlist[i - 1].pmax - unitlist[i - 1].SU) - \
    #                 model.zit[i, t + 1] * max(0, unitlist[i - 1].pmax - unitlist[i - 1].SD)
    #         else:
    #             return model.pit[i, t] <= model.uit[i, t] * unitlist[i - 1].pmax  # 其他情形
    #     else:
    #         return pye.Expression.Skip  # 暂时不太理解何时约束跳过、何时令约束为pi,t<ui,t * Pmax  答：约束里有索引t+1,所以要判断t+1 in range(0,T)
    #
    # model.output_less2 = pye.Constraint(model.Nrange, model.Trange, rule=output_less2_rule)
    #
    #
    # # constraint(23):机组出力上限约束(23)
    # def output_less3_rule(model, i, t):
    #     if i in Se1:
    #         if t + 1 in range(0, T):
    #             return model.pit[i, t] <= model.uit[i, t] * unitlist[i - 1].pmax - model.zit[i, t + 1] * (
    #                     unitlist[i - 1].pmax - unitlist[i - 1].SD) - \
    #                 model.yit[i, t] * max(0, unitlist[i - 1].SD - unitlist[i - 1].SU)
    #         else:
    #             return model.pit[i, t] <= model.uit[i, t] * unitlist[i - 1].pmax  # 其他情形
    #     else:
    #         return pye.Expression.Skip  # 暂时不太理解何时约束跳过、何时令约束为pi,t<ui,t * Pmax  答：约束里有索引t+1,所以要判断t+1 in range(0,T)
    #
    # model.output_less3 = pye.Constraint(model.Nrange, model.Trange, rule=output_less3_rule)


    # 写法2：邓俊师兄论文的约束11
    # 机组出力上下限
    def UnitPlimit_con(model, i, t, type):
        if type == 1:
            return model.pit[i, t] <= model.uit[i, t] * unitlist[i - 1].pmax
        else:
            return model.pit[i, t] >= model.uit[i, t] * unitlist[i - 1].pmin

    model.UnitPlimit_con = pye.Constraint(model.Nrange, model.Trange, pye.RangeSet(1, 2), rule=UnitPlimit_con)


    # 爬坡约束：邓俊师兄论文的约束12、13
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
        # 写法1：涛哥写法,认为start=0.7*ramp up 备注：跟涛哥讨论后,得到的结论是starup必须>=pmin,否则无法保证p0=0,p1>=pmin的情形
        # if (type == 1):
        #     return model.pit[i, t] - ptjy <= utjy * unitlist[i - 1].RU + model.yit[i, t] * unitlist[i - 1].RU * 0.7
        # else:
        #     return ptjy - model.pit[i, t] <= model.uit[i, t] * unitlist[i - 1].RD + model.zit[i, t] * unitlist[i - 1].RD * 0.7
        # 写法2：卢哥写法,认为start=pmin
        # if (type == 1):
        #     return model.pit[i, t] - ptjy <= utjy * unitlist[i - 1].RU + model.yit[i, t] * unitlist[i - 1].pmin
        # else:
        #     return ptjy - model.pit[i, t] <= model.uit[i, t] * unitlist[i - 1].RD + model.zit[i, t] * unitlist[i - 1].pmin
        # 写法3: 使用IEEE118的start up 和 shut dowm功率
        if (type == 1):
            return model.pit[i, t] - ptjy <= utjy * unitlist[i - 1].RU + model.yit[i, t] * unitlist[i - 1].SU
        else:
            return ptjy - model.pit[i, t] <= model.uit[i, t] * unitlist[i - 1].RD + model.zit[i, t] * unitlist[i - 1].SD

    model.PupLimit_con = pye.Constraint(model.Nrange, model.Trange, pye.RangeSet(1, 2), rule=PupLimit_con)


    # 逻辑约束：邓俊师兄论文的约束14
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


    # 最小启停时间约束：邓俊师兄论文的约束15、16
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
                return model.uit[i, t] == 1  # 时间段在仍需连续运行的时间内则是必开的
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
                return model.uit[i, t] == 0  # 时间段在仍需连续停机的时间内则是必关的

    model.minon_con = pye.Constraint(model.Nrange, model.Trange, pye.RangeSet(1, 2), rule=minon_con)


    # 储储
    # 储能充放电功率约束：杨老师课题组论文"精准计及大规模储能电池寿命的电力系统经济调度_李中浩"的约束7、8
    def pstoreLimit_rule(model, i, t, type):
        if type==1:
            return model.pstorec[i, t] <= model.ustorec[i, t] * storelist[i-1].pmax
        else:
            return model.pstored[i, t] <= model.ustored[i, t] * storelist[i-1].pmax

    model.pstoreLimit = pye.Constraint(model.Srange, model.Trange, pye.RangeSet(1, 2), rule=pstoreLimit_rule)


    # 储能充放电状态约束：胡泽春“考虑储能参与调频的风储联合运行优化策略”的约束8
    def pstoreState_rule(model, i, t):
        return model.ustorec[i, t] + model.ustored[i, t] == 1

    model.pstoreState = pye.Constraint(model.Srange, model.Trange, rule=pstoreState_rule)


    # 储能电量平衡约束：杨老师课题组论文"精准计及大规模储能电池寿命的电力系统经济调度_李中浩"的约束10 备注：原文中已经证明，删去约束9(非线性约束)不影响最优解
    def pstoreQ1_rule(model, i, t):
        EB = 0
        if t==1:
            EB = storelist[i-1].Q0
        else:
            EB = model.pstoreEB[i, t-1]
        return model.pstoreEB[i, t] == EB + (storelist[i - 1].etaC * model.pstorec[i, t]
                                             - model.pstored[i, t] / storelist[i - 1].etaD) * (period_length / 60)

    model.pstoreQ1 = pye.Constraint(model.Srange, model.Trange, rule=pstoreQ1_rule)


    # 储能电量上下限约束、储能寿命：杨老师课题组论文"精准计及大规模储能电池寿命的电力系统经济调度_李中浩"的约束11、12
    def pstoreQ2_rule(model, i, t):
        return (storelist[i - 1].Qmin, model.pstoreEB[i, t], storelist[i - 1].Qmax * storelist[i-1].Clife)

    model.pstoreQ2 = pye.Constraint(model.Srange, model.Trange, rule=pstoreQ2_rule)

    # 储能末时段电量约束：第三届求解器大赛的储能终止容量约束
    def pstoreQ3_rule(model, i, t):
        return model.pstoreEB[i, t] >= storelist[i - 1].Q0

    model.pstoreQ3 = pye.Constraint(model.Srange, model.Trange, rule=pstoreQ3_rule)


    # 线路潮流计算式：参考馨茹的计算交流联络线功率的代码,原理：功率分布转移因子法的直流潮流模型
    def CalPline_rule(model, lineNo, t):
        sum = 0
        for busi in range(0, len(buslist)):
            if Punitlist[busi] != -1:
                for i in range(0, len(Punitlist[busi])):
                    sum += model.pit[Punitlist[busi][i], t] * float(PTDF_dict[str(busi + 1)][lineNo - 1])  # 火电出力*功率分布转移因子  #备注: busi+1表示第busi+1个节点,lineNo-1表示第lineNo条线路
            # 储储
            if Pstorelist[busi] != -1:
                for i in range(0, len(Pstorelist[busi])):
                    sum += (model.pstored[Pstorelist[busi][i], t] - model.pstorec[Pstorelist[busi][i], t]) * float(
                        PTDF_dict[str(busi + 1)][lineNo - 1])  # storage出力(放-充)*功率分布转移因子

            sum -= buslist[busi].LoadP * load[t - 1].LoadRate * float(
                PTDF_dict[str(busi + 1)][lineNo - 1])  # busload*功率分布转移因子
        return model.Pline[lineNo, t] == sum

    # model.CalPline = pye.Constraint(model.line, model.Trange, rule=CalPline_rule)                                  # 对整表的线路检索并建立pline等式
    # model.CalPline = pye.Constraint(pye.RangeSet(1, 4), model.Trange, rule=CalPline_rule)                    # 只对特定的几条线路建立pline等式  case2383无阻塞
    model.CalPline = pye.Constraint(pye.RangeSet(1, 20), model.Trange, rule=CalPline_rule)                         # 只对特定的几条线路建立pline等式  case1888设置阻塞


    # 线路潮流约束：参考馨茹的计算交流联络线功率的代码,原理：功率分布转移因子法的直流潮流模型
    def Plinelimit_rule(model, lineNo, t):
        if linelist[lineNo - 1].IsbuildPlinelimit == 'YES':  # 只对应建立安全约束的线路建立安全约束
            # count = model.currentline()
            # progress = count / (PlinelimitCount * T) * 100
            # formatted_progress = f"{progress:.4f}"
            # print(f"当前安全约束建立进度：{formatted_progress}%")
            # model.currentline.set_value(count + 1)
            return (linelist[lineNo - 1].Pmin, model.Pline[lineNo, t],
                    linelist[lineNo - 1].Pmax)  # 同一条线路,列表linelist中的索引等于变量Pline的索引减1
        else:
            return pye.Constraint.Skip

    model.Plinelimit = pye.Constraint(model.line, model.Trange, rule=Plinelimit_rule)


    # 线路潮流约束：参考馨茹的计算交流联络线功率的代码,原理：功率分布转移因子法的直流潮流模型
    # def Plinelimit_rule(model, lineNo, t):
    #     if linelist[lineNo - 1].IsbuildPlinelimit == 'YES':  # 只对应建立安全约束的线路建立安全约束
    #         # count = model.currentline()
    #         # progress = count / (PlinelimitCount * T) * 100
    #         # formatted_progress = f"{progress:.4f}"
    #         # print(f"当前安全约束建立进度: {formatted_progress}%")
    #         # print(f"当前正在建立的安全约束序号为: {count}, 对应线路序号为: {lineNo}, 时段序号为: {t}")
    #         # model.currentline.set_value(count + 1)
    #         sum = 0
    #         for busi in range(0, len(buslist)):
    #             if Punitlist[busi] != -1:
    #                 for i in range(0, len(Punitlist[busi])):
    #                     sum += model.pit[Punitlist[busi][i], t] * float(PTDF_dict[str(busi + 1)][lineNo - 1])  # 火电出力*功率分布转移因子  #备注: busi+1表示第busi+1个节点,lineNo-1表示第lineNo条线路
    #
    #             # 储储
    #             if Pstorelist[busi] != -1:
    #                 for i in range(0, len(Pstorelist[busi])):
    #                     sum += (model.pstored[Pstorelist[busi][i], t] - model.pstorec[Pstorelist[busi][i], t]) * float(
    #                         PTDF_dict[str(busi + 1)][lineNo - 1])  # storage出力(放-充)*功率分布转移因子
    #
    #             sum -= buslist[busi].LoadP * load[t - 1].LoadRate * float(
    #                 PTDF_dict[str(busi + 1)][lineNo - 1])  # busload*功率分布转移因子
    #         return (linelist[lineNo - 1].Pmin, sum,
    #                 linelist[lineNo - 1].Pmax)  # 同一条线路,列表linelist中的索引等于变量Pline的索引减1
    #     else:
    #         return pye.Constraint.Skip
    #
    # model.Plinelimit = pye.Constraint(model.line, model.Trange, rule=Plinelimit_rule)

# endregion


    # print("outing")
    # 保存模型的mps或lp文件
    # example: model.write("your generation path\\XXX.mps", "mps", None,{"symbolic_solver_labels": True})
    # model.write(Dir1 + "\\case300haveStorage.lp", "lp", None, {"symbolic_solver_labels": True})
    # model.write(Dir1 + "\\IEEE118_.lp", "lp", None, {"symbolic_solver_labels": True})
    milp_name = f"{Dir2}.lp"
    milp_path = os.path.join(Dir1, milp_name)
    model.write(milp_path, "lp", None, {"symbolic_solver_labels": True})
    # model.write(Dir1 + "\\case6515rte.lp", "lp", None, {"symbolic_solver_labels": True})






