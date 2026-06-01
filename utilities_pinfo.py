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
        if(self.iniP>0):
            G=self.minontime-self.iniT
            if(G<0):
                return 0
            else:
                return G
        else:
            return 0
    def L(self):
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
        self.avgcost = 0
        self.iniT = 0
        self.t = [calscoreformu1(i) for i in range(1, 25)]
        self.keepOffT = 0
        self.keepOnT = 0
        self.avgcost_scale = 0
        self.hotstartcost_scale = 0
        self.coldstartcost_scale = 0
        self.iniT_scale = 0
        self.t_scale = [calscoreformu1(i/12.5) for i in range(1, 25)]
        self.keepT_scale = 0
        self.beyondT_scale = 0

    def addfenduan(self,l,r,v):
        self.fenduan_num+=1
        self.fenduan_left.append(l)
        self.fenduan_right.append(r)
        self.fenduan_V.append(v)

    def calavgcost(self):
        tmp1 = [ self.fenduan_V[m] * (self.fenduan_right[m]-self.fenduan_left[m]) for m in range(0, self.fenduan_num) ]
        self.avgcost = (self.lowprice + sum(tmp1)) / self.pmax


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
            if Hunit.iniState == 0:
                Hunit.keepOffT = (Hunit.minofftime - abs(Hunit.iniT) > 0) if (Hunit.minofftime - abs(Hunit.iniT) > 0) else 0
            else:
                Hunit.keepOnT  = (Hunit.minontime - abs(Hunit.iniT) > 0) if (Hunit.minontime - abs(Hunit.iniT) > 0) else 0
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


def getPlineTrueIndex(linelist, Blocklinecsvpath):
    """
        Parameters
        ----------
        linelist :
        Blocklinecsvpath :

        output
        ----------
        PlineTruelist :
    """
    line_dict = {(line.Ni, line.Nj): idx for idx, line in enumerate(linelist)}  # Note: Here, .Ni and .Nj must match the Lineparam class in IEEE_g.py; otherwise, an error will occur stating that this attribute does not exist!

    PlineTrueIndex = []
    fp = open(Blocklinecsvpath, "r")
    linecontext = fp.readlines()
    fp.close()

    for index in linecontext:
        list = index.split(',')
        if (list[0].isnumeric()):
            i, j = int(list[1]), int(list[2])
            ture_index = line_dict.get((i, j), -1)
            if ture_index == -1:
                print(f"Warning: Line (i={i}, j={j}) not found in linelist")
            PlineTrueIndex.append(ture_index)

    return PlineTrueIndex


def calscoreformu1(x):
    try:
        x_float = float(x)
        return math.exp(-x_float) / (1 + math.exp(-x_float))
    except (TypeError, ValueError) as e:
        print(f"{x} cannot be converted to a floating-point number {e}...")
        return x

def calscoreformu2(x):
    try:
        x_float = float(x)
        return x_float / (1 + x_float)
    except (TypeError, ValueError) as e:
        print(f"{x} cannot be converted to a floating-point number {e}...")
        return x

class matchVarId:
    def __init__(self, name, genid, t):
        self.name = name
        self.genid = genid
        self.t = t

def parse_candidate_vars(candidate_vars_name):
    pattern = r'^t_(uit|yit|zit|ycoldit|ustorec|ustored)\((\d+)_(\d+)\)$'
    result = []
    for var in candidate_vars_name:
        match = re.match(pattern, var.strip())  # Note: type(var) = str !
        if match:
            x, y, z = match.groups()
            result.append(matchVarId(name=x, genid=int(y), t=int(z)))
        else:
            result.append(matchVarId(name=0, genid=0, t=0))
    return result

class SCIPResultExtractor:

    DEFAULT_PATTERNS = {
        'uit': r'uit\((\d+)_(\d+)\)',
        'pit': r'pit\((\d+)_(\d+)\)',
        'yit': r'yit\((\d+)_(\d+)\)',
        'zit': r'zit\((\d+)_(\d+)\)',
        'ycoldit': r'ycoldit\((\d+)_(\d+)\)',
        'Pline': r'Pline\((\d+)_(\d+)\)',
        'ustorec': r'ustorec\((\d+)_(\d+)\)',
        'ustored': r'ustored\((\d+)_(\d+)\)',
        'pstorec': r'pstorec\((\d+)_(\d+)\)',
        'pstored': r'pstored\((\d+)_(\d+)\)',
    }

    def __init__(self, patterns: Dict[str, str] = None):
        """
        :param patterns
        """
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self._compiled_patterns = {k: re.compile(v) for k, v in self.patterns.items()}

    def extract(self, model: Model, output_xlsx: str='') -> Dict[str, Dict[str, Dict[int, float]]]:
        if model.getObjVal() is None:
            print(f"Problem: {model.getProbName()}. There is no primal solution. Skip......")
            return {var_type: {} for var_type in self._compiled_patterns}

        results = {var_type: {} for var_type in self._compiled_patterns}
        all_vars = model.getVars()

        for var in all_vars:
            var_name = var.name
            for var_type, pattern in self._compiled_patterns.items():
                match = pattern.match(var_name)
                if match:
                    unit_id, time_str = match.groups()
                    time_id = int(time_str)
                    unit_id = str(unit_id)

                    if unit_id not in results[var_type]:
                        results[var_type][unit_id] = {}
                    results[var_type][unit_id][time_id] = model.getVal(var)
                    break

        # return results   # ljm debug +++++++++++++++++++++++++++++++++++++
        dfs = {}
        for var_type, data in results.items():
            df = pd.DataFrame.from_dict({int(unit_id): pd.Series(data[unit_id]) for unit_id in data},orient='index')
            df = df.sort_index(axis=0).sort_index(axis=1, key=lambda x: x.astype(int))
            df.index.name = 'Unit/Plant ID'
            df.columns.name = 'Time Period'
            dfs[var_type] = df

        scip_version = model.version()
        model_name = model.getProbName()
        status = model.getStatus()
        stime = model.getSolvingTime()
        gap_100percent = str(format(100 * model.getGap(), '.2')) + '%'
        model_primal =model.getPrimalbound()
        model_dual = model.getDualbound()
        nnodes = model.getNNodes()
        nlps = model.getNLPs()
        info_dict = {
            "Metric": ["SCIP Version", "Model Name", "Model Status", "Model Runtime","Gap%", "Objective Value", "Dual Bound", 'Total Nodes', 'Total LPs'],
            "Value": [scip_version, model_name, status, stime, gap_100percent, model_primal, model_dual, nnodes, nlps]
        }
        info_df = pd.DataFrame(info_dict)

        with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
            info_df.to_excel(writer, sheet_name='Solve Info', index=False)
            for var_type, df in dfs.items():
                df.to_excel(writer, sheet_name=var_type)

        print(f"\tResults saved to {output_xlsx}")
        return results
        # return results   # ljm debug +++++++++++++++++++++++++++++++++++++

    def update_patterns(self, new_patterns: Dict[str, str]):
        """Dynamic update of matching mode"""
        self.patterns.update(new_patterns)
        self._compiled_patterns = {k: re.compile(v) for k, v in self.patterns.items()}

def load_done_milp_paths(base_path, csv_path):
    """
    Args:
        base_path (str): e.g. case118_1、24GX_750
        csv_path (str):
    Returns:
        list: full csv path list
    """
    done_file_list = []
    with open(csv_path, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                filename = row[0].strip()
                full_path = os.path.join(base_path, filename, f'{filename}.lp')
                if filename and os.path.isfile(full_path):
                    done_file_list.append(filename)
    return done_file_list

