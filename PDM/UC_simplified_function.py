### Unit commitment function: simpify UC version in SCUS model 3
### Author: Jin Lu
### Edited by: Ali Ghaffari
### run UC twice: second run will obtain electricity price

#%%
### Import
from pyomo.environ import *
#import pyomo.environ as pyo
import pickle
import xlsxwriter
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import pandas as pd
import traceback
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
solvername='glpk'
solverpath_folder='C:\\Packages\\glpk-4.65\\w64'                                #does not need to be directly on c drive
solverpath_exe='C:\\Packages\\glpk-4.65\\w64\\glpsol.exe'                       #does not need to be directly on c drive
sys.path.append(solverpath_folder)

## function to find the solar gen&bus list
def find_solar_bus(case_inst):
    solar_genidx_list = []
    solar_busnum_list = []
    for i in range(case_inst.gentotnum):
        if case_inst.gen.fuel_type[i] == 'Solar':  # the solar power plants
            solar_genidx_list.append(i)  # solar gen index in the Gen
            bus_num = case_inst.gen.bus[i]
            solar_busnum_list.append(bus_num)
    return solar_genidx_list, solar_busnum_list

## function to find the wind gen&bus list
def find_wind_bus(case_inst):
    wind_genidx_list = []
    wind_busnum_list = []
    for i in range(case_inst.gentotnum):
        if case_inst.gen.fuel_type[i] == 'Wind':  # the wind power plants
            wind_genidx_list.append(i)  # windgen index in the Gen
            bus_num = case_inst.gen.bus[i]
            wind_busnum_list.append(bus_num)
    return wind_genidx_list, wind_busnum_list

# function to build the Texas UC case (return pyomo case instance)
def build_UC_Texas(UC_pyomo_data):
    ### Python set
    # 24 hour time set
    Time_24 = []
    for i in range(24):
        Time_24.append(i + 1)
    # hour time set (except first hour)
    Time_23 = []
    for i in range(23):
        Time_23.append(i + 2)  # range is 2-24

    ### The pyomo abstract model for UC
    model = AbstractModel()

    # Set
    model.BUS = Set()
    model.LINE = Set()
    model.GEN = Set()
    model.TIME = Set()
    model.BUS_TIME = Set(dimen=2)

    ## Param
    # Bus param
    model.bus_num = Param(model.BUS)
    # Gen param
    model.gen_num = Param(model.GEN)
    model.gen_bus = Param(model.GEN)
    model.gen_cost_P = Param(model.GEN)
    model.gen_cost_NL = Param(model.GEN)
    model.gen_cost_SU = Param(model.GEN)
    model.gen_Pmin = Param(model.GEN)
    model.gen_Pmax = Param(model.GEN)
    model.gen_r10 = Param(model.GEN)
    model.gen_rr = Param(model.GEN)
    # Line param
    model.line_num = Param(model.LINE)
    model.line_x = Param(model.LINE)
    model.line_Pmax = Param(model.LINE)
    model.line_fbus = Param(model.LINE)
    model.line_tbus = Param(model.LINE)
    # Time param
    model.time_num = Param(model.TIME)
    # Bus_Time param
    model.bustime_num = Param(model.BUS_TIME)
    model.load_b_t = Param(model.BUS_TIME)

    # Variable
    # Gen_Time Var
    model.p_g_t = Var(model.GEN, model.TIME)
    model.u_g_t = Var(model.GEN, model.TIME, domain=Binary)
    model.v_g_t = Var(model.GEN, model.TIME)  # v_g_t is simplified to non-binary variable
    model.r_g_t = Var(model.GEN, model.TIME)
    # Line_Time Var
    model.theta_k_t = Var(model.LINE, model.TIME)
    model.p_k_t = Var(model.LINE, model.TIME)
    # Bus_Time Var
    model.theta_b_t = Var(model.BUS, model.TIME)
    model.ldsd_b_t = Var(model.BUS, model.TIME, domain=NonNegativeReals)
    # Simplified renewable curtailment Var
    model.rnwcur_b_t = Var(model.BUS, model.TIME, domain=NonNegativeReals)

    ## Objective function
    def objfunction(model):
        obj = sum(
            model.gen_cost_P[g] * model.p_g_t[g, t] + model.gen_cost_NL[g] * model.u_g_t[g, t] + model.gen_cost_SU[g] *
            model.v_g_t[g, t] for g in model.GEN for t in model.TIME)+1000000*sum(model.ldsd_b_t[b,t] for b in model.BUS for t in model.TIME)
        return obj

    model.object = Objective(rule=objfunction, sense=minimize)

    # Renewable curtailment constraint
    def rnwcur_f_1(model, b, t):
        if model.load_b_t[b,t] >= 0:
            return model.rnwcur_b_t[b, t] == 0
        else:
            return model.rnwcur_b_t[b, t] <= -model.load_b_t[b,t]
    model.rnwcur_cons_1 = Constraint(model.BUS, model.TIME, rule=rnwcur_f_1)

    # Load shedding constraints
    def ldsd_f(model, b, t):
        if model.load_b_t[b,t] >= 0:
            return model.ldsd_b_t[b, t] <= model.load_b_t[b,t]
        else:
            return model.ldsd_b_t[b, t] == 0
    model.ldsd_cons = Constraint(model.BUS, model.TIME, rule=ldsd_f)

    ## v_g_t constraint
    # v_g_t constraint 1
    def vgt_f_1(model, g, t):
        return model.v_g_t[g, t] >= 0

    model.vgt_cons_1 = Constraint(model.GEN, model.TIME, rule=vgt_f_1)

    # v_g_t constraint 2
    def vgt_f_2(model, g, t):
        return model.v_g_t[g, t] <= 1

    model.vgt_cons_2 = Constraint(model.GEN, model.TIME, rule=vgt_f_2)

    ## u_g_t constraint
    def ugt_f(model, g, t):
        return model.u_g_t[g, t] == 1

    model.ugt_cons = Constraint(model.GEN, model.TIME, rule=ugt_f)

    ## Generator power and reserve constraints
    # # P_g_t minimum constraint
    def gen_Pmin_f(model, g, t):
        return model.gen_Pmin[g] * model.u_g_t[g, t] <= model.p_g_t[g, t]

    model.gen_Pmin_cons = Constraint(model.GEN, model.TIME, rule=gen_Pmin_f)

    # P_g_t maximum constraint:
    def gen_Pmax_f(model, g, t):
        return model.p_g_t[g, t] + model.r_g_t[g, t] <= model.gen_Pmax[g] * model.u_g_t[g, t]

    model.gen_Pmax_cons = Constraint(model.GEN, model.TIME, rule=gen_Pmax_f)

    # r_g_t ramping constraint 1:
    def reserve_rr1_f(model, g, t):
        return model.r_g_t[g, t] <= model.gen_r10[g] * model.u_g_t[g, t]

    model.reserve_rr1_cons = Constraint(model.GEN, model.TIME, rule=reserve_rr1_f)

    # r_g_t ramping constraint 2:
    def reserve_rr2_f(model, g, t):
        return model.r_g_t[g, t] >= 0

    model.reserve_rr2_cons = Constraint(model.GEN, model.TIME, rule=reserve_rr2_f)

    # total reserve constraint
    def reserve_tot_f(model, g, t):
        reserve_tot_left = sum(model.r_g_t[g_1, t] for g_1 in model.GEN)
        reserve_tot_right = model.p_g_t[g, t] + model.r_g_t[g, t]
        return reserve_tot_left >= reserve_tot_right

    model.reserve_tot_cons = Constraint(model.GEN, model.TIME, rule=reserve_tot_f)

    # Theta define constraint
    def theta_def_f(model, k, t):
        fbus_num = model.line_fbus[k]
        tbus_num = model.line_tbus[k]
        return model.theta_k_t[k, t] == model.theta_b_t[fbus_num, t] - model.theta_b_t[tbus_num, t]
    model.theta_def_cons = Constraint(model.LINE, model.TIME, rule=theta_def_f)

    # Power flow constraint
    def pf_theta_f(model, k, t):
        return model.p_k_t[k, t] == model.theta_k_t[k, t] / model.line_x[k]
    model.pf_theta_cons = Constraint(model.LINE, model.TIME, rule=pf_theta_f)

    # Power flow min constraint:
    def pf_min_f(model, k, t):
        return model.p_k_t[k, t] >= -1 * model.line_Pmax[k]  # relax line rating

    model.pf_min_cons = Constraint(model.LINE, model.TIME, rule=pf_min_f)

    # Power flow max constraint:
    def pf_max_f(model, k, t):
        return model.p_k_t[k, t] <= 1 * model.line_Pmax[k]  # relax line rating

    model.pf_max_cons = Constraint(model.LINE, model.TIME, rule=pf_max_f)

    # Nodal balance constraint
    def nodal_balance_f(model, b, t):
        nodal_balance_left = sum(model.p_g_t[g, t] for g in model.GEN if model.gen_bus[g] == b)
        nodal_balance_left += sum(model.p_k_t[k, t] for k in model.LINE if model.line_tbus[k] == b)
        nodal_balance_left -= sum(model.p_k_t[k, t] for k in model.LINE if model.line_fbus[k] == b)
        nodal_balance_right = model.load_b_t[b, t] + model.rnwcur_b_t[b,t]-model.ldsd_b_t[b,t]
        return nodal_balance_left == nodal_balance_right

    model.nodal_balance_cons = Constraint(model.BUS, model.TIME, rule=nodal_balance_f)

    # Generator ramping rate constraint 1
    # Assume normal/startup/shutdown ramping rates are the same
    def gen_rr1_f(model, g, t):
        t_previous = model.time_num[t] - 1
        return model.p_g_t[g, t] - model.p_g_t[g, t_previous] <= model.gen_rr[g]

    model.gen_rr1_cons = Constraint(model.GEN, Time_24, rule=gen_rr1_f)

    # Generator ramping rate constraint 2
    def gen_rr2_f(model, g, t):
        t_previous = model.time_num[t] - 1
        return model.p_g_t[g, t] - model.p_g_t[g, t_previous] >= -model.gen_rr[g]

    model.gen_rr2_cons = Constraint(model.GEN, Time_24, rule=gen_rr2_f)

    # Variable V constraint
    def var_v_f(model, g, t):
        if t == 0:
            return model.v_g_t[g, t] >= 0  # no v_g_t constraint for t=0
        else:
            return model.v_g_t[g, t] >= model.u_g_t[g, t] - model.u_g_t[g, t - 1]

    model.var_v_cons = Constraint(model.GEN, model.TIME, rule=var_v_f)

    # load case data and create instance
    # print('start creating the instance')
    case_pyomo = model.create_instance(UC_pyomo_data)
    # dual variable setting
    case_pyomo.dual = pyomo.environ.Suffix(direction=pyomo.environ.Suffix.IMPORT)
    # print('finish creating the instance')
    # case_pyomo.pprint()
    return case_pyomo


    ### Solve and solution

# function to build the Texas UC case with fixed u_g_t (return pyomo case instance)
def build_UC_Texas_Run2(UC_pyomo_data,UC_case):
    ### Python set
    # 24 hour time set
    Time_24 = []
    for i in range(24):
        Time_24.append(i + 1)
    # 23 hour time set
    Time_23 = []
    for i in range(23):
        Time_23.append(i + 2)

    ### The pyomo abstract model for UC
    model = AbstractModel()

    # Set
    model.BUS = Set()
    model.LINE = Set()
    model.GEN = Set()
    model.TIME = Set()
    model.BUS_TIME = Set(dimen=2)

    ## Param
    # Bus param
    model.bus_num = Param(model.BUS)
    # Gen param
    model.gen_num = Param(model.GEN)
    model.gen_bus = Param(model.GEN)
    model.gen_cost_P = Param(model.GEN)
    model.gen_cost_NL = Param(model.GEN)
    model.gen_cost_SU = Param(model.GEN)
    model.gen_Pmin = Param(model.GEN)
    model.gen_Pmax = Param(model.GEN)
    model.gen_r10 = Param(model.GEN)
    model.gen_rr = Param(model.GEN)
    # Line param
    model.line_num = Param(model.LINE)
    model.line_x = Param(model.LINE)
    model.line_Pmax = Param(model.LINE)
    model.line_fbus = Param(model.LINE)
    model.line_tbus = Param(model.LINE)
    # Time param
    model.time_num = Param(model.TIME)
    # Bus_Time param
    model.bustime_num = Param(model.BUS_TIME)
    model.load_b_t = Param(model.BUS_TIME)

    # Variable
    # Gen_Time Var
    model.p_g_t = Var(model.GEN, model.TIME)
    # model.u_g_t = Var(model.GEN, model.TIME, domain=Binary)
    model.v_g_t = Var(model.GEN, model.TIME)  # v_g_t is simplified to non-binary variable
    model.r_g_t = Var(model.GEN, model.TIME)
    # Line_Time Var
    model.theta_k_t = Var(model.LINE, model.TIME)
    model.p_k_t = Var(model.LINE, model.TIME)
    # Bus_Time Var
    model.theta_b_t = Var(model.BUS, model.TIME)
    model.ldsd_b_t = Var(model.BUS, model.TIME, domain=NonNegativeReals)
    # Renewable curtailment Var
    model.rnwcur_b_t = Var(model.BUS, model.TIME, domain=NonNegativeReals)

    ## Objective function
    def objfunction(model):
        obj = sum(
            model.gen_cost_P[g] * model.p_g_t[g, t] + model.gen_cost_NL[g] * UC_case.u_g_t[g, t]() + model.gen_cost_SU[g] *
            model.v_g_t[g, t] for g in model.GEN for t in model.TIME)+1000000*sum(model.ldsd_b_t[b,t] for b in model.BUS for t in model.TIME)
        return obj

    model.object = Objective(rule=objfunction, sense=minimize)

    # Renewable curtailment constraint
    def rnwcur_f_1(model, b, t):
        if model.load_b_t[b,t] >= 0:
            return model.rnwcur_b_t[b, t] == 0
        else:
            return model.rnwcur_b_t[b, t] <= -model.load_b_t[b,t]

    model.rnwcur_cons_1 = Constraint(model.BUS, model.TIME, rule=rnwcur_f_1)

    # Load shedding constraints
    def ldsd_f(model, b, t):
        if model.load_b_t[b,t] >= 0:
            return model.ldsd_b_t[b, t] <= model.load_b_t[b,t]
        else:
            return model.ldsd_b_t[b, t] == 0
    model.ldsd_cons = Constraint(model.BUS, model.TIME, rule=ldsd_f)

    ## v_g_t constraint
    # v_g_t constraint 1
    def vgt_f_1(model, g, t):
        return model.v_g_t[g, t] >= 0

    model.vgt_cons_1 = Constraint(model.GEN, model.TIME, rule=vgt_f_1)

    # v_g_t constraint 2
    def vgt_f_2(model, g, t):
        return model.v_g_t[g, t] <= 1

    model.vgt_cons_2 = Constraint(model.GEN, model.TIME, rule=vgt_f_2)

    ## u_g_t constraint
    # def ugt_f(model, g, t):
    #     return model.u_g_t[g, t] == 1
    # model.ugt_cons = Constraint(model.GEN, model.TIME, rule=ugt_f)

    ## Generator power and reserve constraints
    # # P_g_t minimum constraint
    def gen_Pmin_f(model, g, t):
        return model.gen_Pmin[g] * UC_case.u_g_t[g, t]() <= model.p_g_t[g, t]

    model.gen_Pmin_cons = Constraint(model.GEN, model.TIME, rule=gen_Pmin_f)

    # P_g_t maximum constraint:
    def gen_Pmax_f(model, g, t):
        return model.p_g_t[g, t] + model.r_g_t[g, t] <= model.gen_Pmax[g] * UC_case.u_g_t[g, t]()

    model.gen_Pmax_cons = Constraint(model.GEN, model.TIME, rule=gen_Pmax_f)

    # r_g_t ramping constraint 1:
    def reserve_rr1_f(model, g, t):
        return model.r_g_t[g, t] <= model.gen_r10[g] * UC_case.u_g_t[g, t]()

    model.reserve_rr1_cons = Constraint(model.GEN, model.TIME, rule=reserve_rr1_f)

    # r_g_t ramping constraint 2:
    def reserve_rr2_f(model, g, t):
        return model.r_g_t[g, t] >= 0

    model.reserve_rr2_cons = Constraint(model.GEN, model.TIME, rule=reserve_rr2_f)

    # total reserve constraint
    def reserve_tot_f(model, g, t):
        reserve_tot_left = sum(model.r_g_t[g_1, t] for g_1 in model.GEN)
        reserve_tot_right = model.p_g_t[g, t] + model.r_g_t[g, t]
        return reserve_tot_left >= reserve_tot_right

    model.reserve_tot_cons = Constraint(model.GEN, model.TIME, rule=reserve_tot_f)

    # Theta define constraint
    def theta_def_f(model, k, t):
        fbus_num = model.line_fbus[k]
        tbus_num = model.line_tbus[k]
        return model.theta_k_t[k, t] == model.theta_b_t[fbus_num, t] - model.theta_b_t[tbus_num, t]
    model.theta_def_cons = Constraint(model.LINE, model.TIME, rule=theta_def_f)

    # Power flow constraint
    def pf_theta_f(model, k, t):
        return model.p_k_t[k, t] == model.theta_k_t[k, t] / model.line_x[k]
    model.pf_theta_cons = Constraint(model.LINE, model.TIME, rule=pf_theta_f)

    # Power flow min constraint:
    def pf_min_f(model, k, t):
        return model.p_k_t[k, t] >= -1 * model.line_Pmax[k]  # relax line rating

    model.pf_min_cons = Constraint(model.LINE, model.TIME, rule=pf_min_f)

    # Power flow max constraint:
    def pf_max_f(model, k, t):
        return model.p_k_t[k, t] <= 1 * model.line_Pmax[k]  # relax line rating

    model.pf_max_cons = Constraint(model.LINE, model.TIME, rule=pf_max_f)

    # Nodal balance constraint
    def nodal_balance_f(model, b, t):
        nodal_balance_left = sum(model.p_g_t[g, t] for g in model.GEN if model.gen_bus[g] == b)
        nodal_balance_left += sum(model.p_k_t[k, t] for k in model.LINE if model.line_tbus[k] == b)
        nodal_balance_left -= sum(model.p_k_t[k, t] for k in model.LINE if model.line_fbus[k] == b)
        nodal_balance_right = model.load_b_t[b, t] + model.rnwcur_b_t[b,t]-model.ldsd_b_t[b,t]
        return nodal_balance_left == nodal_balance_right

    model.nodal_balance_cons = Constraint(model.BUS, model.TIME, rule=nodal_balance_f)

    # Generator ramping rate constraint 1
    # Assume normal/startup/shutdown ramping rates are the same
    def gen_rr1_f(model, g, t):
        t_previous = model.time_num[t] - 1
        return model.p_g_t[g, t] - model.p_g_t[g, t_previous] <= model.gen_rr[g]

    model.gen_rr1_cons = Constraint(model.GEN, Time_24, rule=gen_rr1_f)

    # Generator ramping rate constraint 2
    def gen_rr2_f(model, g, t):
        t_previous = model.time_num[t] - 1
        return model.p_g_t[g, t] - model.p_g_t[g, t_previous] >= -model.gen_rr[g]

    model.gen_rr2_cons = Constraint(model.GEN, Time_24, rule=gen_rr2_f)

    # Variable V constraint
    def var_v_f(model, g, t):
        if t == 0:
            return model.v_g_t[g, t] >= 0  # no v_g_t constraint for t=0
        else:
            return model.v_g_t[g, t] >= UC_case.u_g_t[g, t]() - UC_case.u_g_t[g, t - 1]()

    model.var_v_cons = Constraint(model.GEN, model.TIME, rule=var_v_f)

    # load case data and create instance
    # print('start creating the instance')
    case_pyomo = model.create_instance(UC_pyomo_data)
    # dual variable setting
    case_pyomo.dual = pyomo.environ.Suffix(direction=pyomo.environ.Suffix.IMPORT)
    # print('finish creating the instance')
    # case_pyomo.pprint()
    return case_pyomo

# function to pyomo solving given a UC case
# def solve_UC(UC_case,pickle_filename,dat_filename):
def solve_UC(UC_case):
    # set the solver
    UC_solver = SolverFactory('glpk',executable=solverpath_exe)
    # UC_solver = SolverFactory('conopt',
    #                               executable='D:\\OneDrive - University Of Houston\\work folder\\ampl new\\ampl_mswin64_1\\conopt.exe')
    UC_solver.options.mipgap = 0.001
    results = UC_solver.solve(UC_case)
    # print('the solution is found')
    # # display solution
    # print("\nresults.Solution.Status: " + str(results.Solution.Status))
    # print("\nresults.solver.status: " + str(results.solver.status))
    # print("\nresults.solver.termination_condition: " + str(results.solver.termination_condition))
    # print("\nresults.solver.termination_message: " + str(results.solver.termination_message) + '\n')

    # save solution to pickle file
    # pickle.dump(results, open(pickle_filename, "wb"))
    # write result
    # write_UCresult_Texas(UC_case, dat_filename)

# function to write the Texas UC result (write to savefile_name file)
def write_UCresult_Texas(UC_case, savefile_name, lshed_dir, recur_dir):
    ### print result
    f = open(savefile_name, 'w')
    # # print u_g_t
    # u_g_t_str = 'u_g_t: \n'
    # f.write(u_g_t_str)
    # for g in UC_case.GEN:
    #     u_g_t_str = ''
    #     for t in UC_case.TIME:
    #         u_g_t_str = u_g_t_str + str(int(UC_case.u_g_t[g, t]())) + ' '
    #     f.write(u_g_t_str)
    #     f.write('\n')
    # f.write('\n')
    # print p_g_t
    p_g_t_str = 'p_g_t: \n'
    f.write(p_g_t_str)
    for g in UC_case.GEN:
        p_g_t_str = ''
        for t in UC_case.TIME:
            p_g_t_str = p_g_t_str + str(int(UC_case.p_g_t[g, t]())) + ' '
        f.write(p_g_t_str)
        f.write('\n')
    f.write('\n')
    # print p_k_t
    p_k_t_str = 'p_k_t: \n'
    f.write(p_k_t_str)
    for k in UC_case.LINE:
        p_k_t_str = ''
        for t in UC_case.TIME:
            p_k_t_str = p_k_t_str + str(int(UC_case.p_k_t[k, t]())) + ' '
        f.write(p_k_t_str)
        f.write('\n')
    f.write('\n')
    # print power flow percentage
    p_k_t_str = 'p_k_t_pct: \n'
    f.write(p_k_t_str)
    for k in UC_case.LINE:
        p_k_t_pct_str = ''
        for t in UC_case.TIME:
            p_k_t = UC_case.p_k_t[k, t]()
            p_k_max = UC_case.line_Pmax[k]
            # print(str(case_pyomo.line_Pmax[k])+' ')
            p_k_t_pct_str = p_k_t_pct_str + str(int(p_k_t / p_k_max * 100)) + '%' + ' '
        f.write(p_k_t_pct_str)
        f.write('\n')
    f.write('\n')
    # print r_g_t
    r_g_t_str = 'r_g_t: \n'
    f.write(r_g_t_str)
    for g in UC_case.GEN:
        r_g_t_str = ''
        for t in UC_case.TIME:
            r_g_t = UC_case.r_g_t[g, t]()
            r_g_t_str += str(r_g_t) + ' '
        f.write(r_g_t_str)
        f.write('\n')
    f.write('\n')
    f.close()
    # print load shedding
    # flnm_ldsd = 'load_shedding.txt'
    flnm_ldsd = os.path.join(lshed_dir, os.path.basename(savefile_name))
    f = open(flnm_ldsd, 'w')
    for b in UC_case.BUS:
        lmp_str = ''
        for t in UC_case.TIME:
            lmp = UC_case.ldsd_b_t[b, t]()
            lmp_str += str(lmp) + ' '
        f.write(lmp_str)
        f.write('\n')
    f.write('\n')
    f.close()
    # print renewable curtailment
    # flnm_ldsd = 'rnw_curtail.txt'
    flnm_ldsd = os.path.join(recur_dir, os.path.basename(savefile_name))
    f = open(flnm_ldsd, 'w')
    for b in UC_case.BUS:
        lmp_str = ''
        for t in UC_case.TIME:
            lmp = UC_case.rnwcur_b_t[b, t]()
            lmp_str += str(lmp) + ' '
        f.write(lmp_str)
        f.write('\n')
    f.write('\n')
    f.close()

### Save the LMP result into the excel file
def saveLMP(UC_case_run2,save_flnm):
    nodal_balance_cons = getattr( UC_case_run2, 'nodal_balance_cons')
    wb = xlsxwriter.Workbook(save_flnm)
    for t in UC_case_run2.TIME:
        sheet_name = 'Hour ' + str(t)
        hour_sheet = wb.add_worksheet(sheet_name)
        hour_sheet.write(0, 0, 'Bus Number')
        hour_sheet.write(0, 1, 'Price') 
        for b in UC_case_run2.BUS:
            hour_sheet.write(b,0,b)
            hour_sheet.write(b,1,UC_case_run2.dual.get(nodal_balance_cons[b,t]))
            # print('bus '+str(b)+' time '+str(t)+' : ')
            # test_dual = UC_case_run2.dual.get(nodal_balance_cons[b,t])
            # print(test_dual)
    wb.close()
    
# =============================================================================
# gen2bus
# =============================================================================
### convert PGMS output to ABM input     
def gen2bus(UC_case, gen_list_c_dir, UC_output, pgms_loads_updated_dir, load, y):
    gen_list_raw = pd.read_csv(os.path.join(gen_list_c_dir,"gen_list_c_" + 
                                            str(y) + ".csv"))                           # read generators list from *.csv                                                 
    ### convert price at generator level to bus level
    nodal_balance_cons = getattr(UC_case, 'nodal_balance_cons')                         # get nodal_balance results from solved case
    bus_price = pd.DataFrame(columns = ["Bus_Number"])                                  # create an empty dataframe to store results
    df = pd.DataFrame()                                                                 # create an empty dataframe to store hourly price values
    for t in UC_case.TIME:                                                              # loop over 24 hours
        price_col = 'Price_' + str(t)                                                   # new column name = price at hour t
        for b in UC_case.BUS:                                                           # loop over buses
            df.loc[b-1, price_col] = UC_case.dual.get(nodal_balance_cons[b,t])          # price at bus b at hour t
    bus_price["Bus_Number"] = np.arange(1, 124)                                         # create a list of 123 bus numbers
    bus_price["Price"] = df.mean(axis=1)                                                # take average of hourly prices to get the daily price
    
    ### convert power output at generator level to bus level
    bus_power = pd.DataFrame(columns = ["Bus_Number", "Power"])                         # create a dataframe to store results
    bus_power["Bus_Number"] = gen_list_raw["gen_bus"]                                   # get bus numbers for generators from gen_list
    p_g_t = UC_case.p_g_t[:,:]()                                                        # extract output power of generators as a list
    bus_power["Power"] = np.reshape(p_g_t, (-1,25)).sum(axis=1)                         # calculate total generator power in 24 hours
    for b in range(1,124,1):                                                            # loop over buses
        if b not in bus_power["Bus_Number"].values:                                     # if bus is not present in generators list
            new_row = [b, 0]                                                            # create a new row with bus number and zero power
            bus_power.loc[len(bus_power)] = new_row                                     # append new_row to the end of dataframe
    bus_power = bus_power.groupby("Bus_Number", as_index=False).sum()                   # sum generators power for each bus
    
    ### add demand for scenario as a new column to result
    df = pd.read_csv(os.path.join(pgms_loads_updated_dir, str(y), load), header=None)   # read load from *.txt file
    bus_load = pd.DataFrame(columns = ["Bus_Number"])                                   # create an empty dataframe to store results
    bus_load["Bus_Number"] = np.arange(1, 124)                                          # create a list of 123 bus numbers
    bus_load["Load"] = df.values.reshape(-1,25).sum(axis=1)                             # sum hourly loads to get the daily load
    
    result = pd.merge(pd.merge(bus_price,bus_power,on='Bus_Number')                     # merge price, power, and load in one dataframe
                      ,bus_load,on='Bus_Number')                                     
    result.to_csv(UC_output, index=False)                                               # save output to *.csv file

# =============================================================================
# pgms    
# =============================================================================
"""run the energy model for i load profiles for year y. this code does not use
parallelization."""
# def pgms(pgms_inputs_dir, pgms_loads_updated_dir, gen_list_dir, pgms_outputs_dir, y):
#     pgms_inputs_dir = os.path.join(pgms_inputs_dir, str(y))                             # input files directory
#     input_files = os.listdir(pgms_inputs_dir)                                           # list input files for different load profiles
#     load_files = os.listdir(os.path.join(pgms_loads_updated_dir, str(y)))               # list load profiles
#     infeasible_loads = []                                                               # create an empty list to store infeasible load scenarios
#     for i in range(len(input_files)):                                                   # loop over load profiles
#         try:                                                                            # if solution is feasible
#             UC_input = input_files[i]                                                   # ith input file for PGMS
#             load = load_files[i]                                                        # ith load file
#             UC_pyomo_data = os.path.join(pgms_inputs_dir, UC_input)                     # create input full file path
#             UC_case_run1 = build_UC_Texas(UC_pyomo_data)                                # build UC Texas model for 1st run
#             solve_UC(UC_case_run1)                                                      # solve the model for the 1st time
#             UC_case_run2 = build_UC_Texas_Run2(UC_pyomo_data, UC_case_run1)             # build UC Texas model for 2nd run
#             solve_UC(UC_case_run2)                                                      # solve the model for the 2nd time
#             if i==0:                                                                    # create directory for outputs
#                 os.mkdir(os.path.join(pgms_outputs_dir, str(y)))
#             UC_output = os.path.join(pgms_outputs_dir, str(y), UC_input[10:-3]+"csv")   # create output full file path
#             gen2bus(UC_case_run2, gen_list_dir, UC_output, 
#                     pgms_loads_updated_dir, load, y)                                    # convert PGMS output to ABM input and save it in *.csv format
#         except TypeError:                                                               # if solution is not feasible
#             infeasible_loads.append(UC_input)                                           # append infeasible load scenario to a list
#     print("Energy model completed run for " + str(y))                                   # print a message to show that simulation is done for year y
    
#     # return infeasible_loads                                                             # return a list of infeasible load scenarios

# =============================================================================
# pgms    
# =============================================================================
### run the energy model for 16 scenarios for year y
def solve_pyomo(pgms_inputs_dir, pgms_loads_updated_dir, gen_list_c_dir, 
                  pgms_outputs_dir, y, i, input_files, load_files, 
                  infeasible_loads):
    try:                                                                                # if solution is feasible
        UC_input = input_files[i]                                                       # ith input file for PGMS
        load = load_files[i]                                                            # ith load file
        UC_pyomo_data = os.path.join(pgms_inputs_dir, UC_input)                         # create input full file path
        UC_case_run1 = build_UC_Texas(UC_pyomo_data)                                    # build UC Texas model for 1st run
        solve_UC(UC_case_run1)                                                          # solve the model for the 1st time
        UC_case_run2 = build_UC_Texas_Run2(UC_pyomo_data, UC_case_run1)                 # build UC Texas model for 2nd run
        solve_UC(UC_case_run2)                                                          # solve the model for the 2nd time
        isExist = os.path.exists(os.path.join(pgms_outputs_dir, str(y)))                # check if directory exists
        ### list of directories to save results
        sol_dir = os.path.join(pgms_outputs_dir, str(y), '01_Solutions')
        price_dir = os.path.join(pgms_outputs_dir, str(y), '02_Prices')
        lshed_dir = os.path.join(pgms_outputs_dir, str(y), '03_load_shedding')
        recur_dir = os.path.join(pgms_outputs_dir, str(y), '04_renewable_curtailment')

        try:
            if isExist==False:
                os.mkdir(os.path.join(pgms_outputs_dir, str(y)))                        # create directory for outputs
                os.mkdir(sol_dir)                                                       # create directory for PGMS solutions
                os.mkdir(price_dir)                                                     # create directory for price values
                os.mkdir(lshed_dir)                                                     # create directory for load shedding results
                os.mkdir(recur_dir)                                                     # create directory for renewable curtailment results

                # print(f'New directory is created for {y}!') 
        except FileExistsError:
            print("Parallelization error.")
        UC_output = os.path.join(price_dir, UC_input[10:-3]+"csv")                      # create output full file path
        gen2bus(UC_case_run2, gen_list_c_dir, UC_output, 
                pgms_loads_updated_dir, load, y)                                        # convert PGMS output to ABM input and save it in *.csv format
        savefile_name = os.path.join(sol_dir, UC_input[10:-3]+'dat')                    # pyomo result filename
        write_UCresult_Texas(UC_case_run2, savefile_name, lshed_dir, recur_dir)
    except TypeError:                                                                   # if solution is not feasible
        infeasible_loads.append(UC_input)                                               # append infeasible load scenario to a list
    # print(f'Energy model completed run for load profile {i}')                           # print a message to show that simulation is done for year y
    
    # return infeasible_loads                                                             # return a list of infeasible load scenarios

# =============================================================================
# pgms
# =============================================================================
"""run the energy model for i load profiles for year y. this code uses
parallelization and therefore, is much faster."""
def pgms(pgms_inputs_dir, pgms_loads_updated_dir, gen_list_c_dir, 
         pgms_outputs_dir, y, n_cores=-1):
    print('Solving PDM for year {}:'.format(y))
    pgms_inputs_dir = os.path.join(pgms_inputs_dir, str(y))                             # input files directory
    input_files = os.listdir(pgms_inputs_dir)                                           # list input files for different load profiles
    load_files = os.listdir(os.path.join(pgms_loads_updated_dir, str(y)))               # list load profiles
    infeasible_loads = []                                                               # create an empty list to store infeasible load scenarios
    
    # parallelize the process by iteration over input files (load profiles)
    Parallel(n_jobs=n_cores)\
        (delayed(solve_pyomo)\
         (pgms_inputs_dir, pgms_loads_updated_dir, gen_list_c_dir, 
                           pgms_outputs_dir, y, i, input_files, load_files, 
                           infeasible_loads) for i in tqdm(range(len(input_files)),miniters=1))
    print('Completed PDM for year {}.\n'.format(y))
    if not os.path.exists(os.path.join(pgms_outputs_dir, str(y))):                      # if no solutions are found
        print('PGMS could not solve dispatch model for any of the profiles.')





