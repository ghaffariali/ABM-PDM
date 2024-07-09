### Unit commitment function: simpify UC version in SCUS model 3
### Author: Jin Lu
### run UC twice: second run will obtain electricity price

### Import
from pyomo.environ import *
#import pyomo.environ as pyo
import pickle
import xlsxwriter
import numpy as np
import os
import matplotlib.pyplot as plt


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
    model.u_g_t = Var(model.GEN, model.TIME, domain=Binary) # u_g_t
    model.v_g_t = Var(model.GEN, model.TIME, domain=Binary)  # v_g_t
    model.r_g_t = Var(model.GEN, model.TIME)
    # Line_Time Var
    model.theta_k_t = Var(model.LINE, model.TIME)
    model.p_k_t = Var(model.LINE, model.TIME)
    # Bus_Time Var
    model.theta_b_t = Var(model.BUS, model.TIME)

    ## Objective function
    def objfunction(model):
        obj = sum(
            model.gen_cost_P[g] * model.p_g_t[g, t] + model.gen_cost_NL[g] * model.u_g_t[g, t] + model.gen_cost_SU[g] *
            model.v_g_t[g, t] for g in model.GEN for t in model.TIME)
        return obj

    model.object = Objective(rule=objfunction, sense=minimize)


    ## u_g_t constraint
    def ugt_f(model, g):
        return model.u_g_t[g, 0] == 0
    model.ugt_cons = Constraint(model.GEN, rule=ugt_f)

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
        nodal_balance_right = model.load_b_t[b, t]
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
    print('start creating the instance')
    case_pyomo = model.create_instance(UC_pyomo_data)
    # dual variable setting
    case_pyomo.dual = pyomo.environ.Suffix(direction=pyomo.environ.Suffix.IMPORT)
    print('finish creating the instance')
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
    model.v_g_t = Var(model.GEN, model.TIME,domain=Binary)  # v_g_t is simplified to non-binary variable
    model.r_g_t = Var(model.GEN, model.TIME)
    # Line_Time Var
    model.theta_k_t = Var(model.LINE, model.TIME)
    model.p_k_t = Var(model.LINE, model.TIME)
    # Bus_Time Var
    model.theta_b_t = Var(model.BUS, model.TIME)

    ## Objective function
    def objfunction(model):
        obj = sum(
            model.gen_cost_P[g] * model.p_g_t[g, t] + model.gen_cost_NL[g] * UC_case.u_g_t[g, t]() + model.gen_cost_SU[g] *
            model.v_g_t[g, t] for g in model.GEN for t in model.TIME)
        return obj

    model.object = Objective(rule=objfunction, sense=minimize)

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
        nodal_balance_right = model.load_b_t[b, t]
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
    print('start creating the instance')
    case_pyomo = model.create_instance(UC_pyomo_data)
    # dual variable setting
    case_pyomo.dual = pyomo.environ.Suffix(direction=pyomo.environ.Suffix.IMPORT)
    print('finish creating the instance')
    # case_pyomo.pprint()
    return case_pyomo

# function to pyomo solving given a UC case
def solve_UC(UC_case,pickle_filename,dat_filename):
    # set the solver
    UC_solver = SolverFactory('glpk')
    # UC_solver = SolverFactory('conopt',
    #                               executable='D:\\OneDrive - University Of Houston\\work folder\\ampl new\\ampl_mswin64_1\\conopt.exe')
    UC_solver.options.mipgap = 0.001
    results = UC_solver.solve(UC_case)
    print('the solution is found')
    # display solution
    print("\nresults.Solution.Status: " + str(results.Solution.Status))
    print("\nresults.solver.status: " + str(results.solver.status))
    print("\nresults.solver.termination_condition: " + str(results.solver.termination_condition))
    print("\nresults.solver.termination_message: " + str(results.solver.termination_message) + '\n')

    # save solution to pickle file
    pickle.dump(results, open(pickle_filename, "wb"))
    # write result
    write_UCresult_Texas(UC_case, dat_filename)

# function to write the Texas UC result (write to savefile_name file)
def write_UCresult_Texas(UC_case, savefile_name):
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

#### Main: Example codes to run Unit commitment for a test case
## first run of UC
UC_pyomo_data = 'formpyomo_UC_data.dat'
UC_case_run1 = build_UC_Texas(UC_pyomo_data)
solve_UC(UC_case_run1,"UCresults_full.pickle",'UCsolution_full.dat')
## Second run of UC
# fixed the u_g_t, solve again and obtain dual variable (electricity price)
UC_case_run2 = build_UC_Texas_Run2(UC_pyomo_data, UC_case_run1)
solve_UC(UC_case_run2,"UCresults_full_run2.pickle","UCsolution_full_run2.dat")
saveLMP(UC_case_run2,'LMP_UC_full.csv')