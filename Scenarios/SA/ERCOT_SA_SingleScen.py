# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:49:06 2022

@author: Ali Ghaffari       alg721@lehigh.edu

Agent-based model 
If IRR>0.08 then the agent will invest, otherwise it won't. Investment amount
is determined by a normal distribution of the existing capacity. Based on data, 
a typical solar system in the U.S. can produce electricity at the cost of $0.06 
to $0.08 per kilowatt-hour. With a payback period of 5.5 years in New Jersey, 
Massachusetts, and California, the ROI is 18.2% annually. In some other states, 
the payback period can be up to 12 years, which translates to an 8.5% annual return.
In this code, risk is static (predetermined by calibration) and does not change
annually.
"""
#%% import libraries and functions
# import libraries
import time; st = time.time()                                                       # import time and measure start time
import sys
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None                                           # turn off warnings (default='warn')
import datetime
# add the followiong to path to import functions
cwd = r'C:\Projects\ABM-PDM'                                                        # current working directory
os.chdir(cwd)                                                                       # change working directory
sys.path.append(r'C:\Projects\ABM-PDM\ABM')                                         # append ABM directory to path
sys.path.append(r'C:\Projects\ABM-PDM\PDM')                                         # append PGMS directory to path                            
# import functions
from functions import pgms2abm, bus2gen, load_average, ZoneInvest, AgentInvest,\
    aggregate, copydirectory, base2input, update_risk, cal_renewable,\
        gen_accumulation
from UC_simplified_function import pgms

#%% date
today = datetime.datetime.now()                                                     # create date object
date_now = today.strftime("%b-%d %H-%M")                                            # save date object as string
print("Today's date:", today)                                                       # output date

#%% read data files
# directories
result_dir = os.path.join(cwd, 'Results', '02_SA', 'ERCOT')                         # result directory
os.mkdir(os.path.join(result_dir, date_now))                                        # create a folder to store results for the date of simulation        
# ABM data
future_data_dir = os.path.join(cwd, 'ABM', 'Future_Data')                           # future data directory
Demand_dir = os.path.join(future_data_dir, "Demand")                                # future demand directory
# Price_dir = os.path.join(future_data_dir, "Price")                                # future price directory
Cost_dir = os.path.join(future_data_dir, "Cost")                                    # future cost directory
CF_dir = os.path.join(future_data_dir, "CF")                                        # future capacity factor directory
Retire_dir = os.path.join(future_data_dir, "Retirement")                            # future retired capacity directory
bus_map_file = os.path.join(future_data_dir,"bus_map.csv")                          # bus maps directory
# temp_data_file = os.path.join(future_data_dir, "future_temp.csv")

"""read market data including demand, capacity factor, capacity, and cost 
Source: EIA-860: 3_1_Generator_Y2020.xlsx, which includes three tabs: Operable,
Proposed, Retired and Canceled """
Annual_Demand = pd.read_csv(os.path.join(Demand_dir, "Future_annual_demand.csv"),
                            usecols=["Year", "Energy (MWh)"], index_col='Year')     # read ERCTO annual load prediction for 2021-2051
cost_file = "Cost_1.00_pect_decrease.pkl"
cf_file = "CF_1.00_pect_increase.pkl"
scen = 'ERCOT_{}cost_{}cf'.format(cost_file[-20:-17], cf_file[-20:-17])             # get the name of SA scenario
Cost = pd.read_pickle(os.path.join(Cost_dir, cost_file))                            # read future Cost data                    
CF = pd.read_pickle(os.path.join(CF_dir, cf_file))                                  # read future CF data                      
Retire = pd.read_pickle(os.path.join(Retire_dir, "Retirement_no_change.pkl"))       # constant retirement
bus_map = pd.read_csv(bus_map_file)                                                 # bus map to convert ABM and energy outputs either format

src = os.path.join(cwd, 'Source_ERCOT_Direct')                                      # source files to run the code
dst = os.path.join(result_dir, date_now, scen)                                      # destination to create new files and results
copydirectory(src, dst)                                                             # copy files in source to destination

#%% model initialization
data = {'NG':[30, Cost.iloc[0,0], 56.6],                                            # save data to a dictionary (lifespan, cost, capacity factor)
        'Solar':[25, Cost.iloc[0,1], 24.9],                                         # Solar ($/MW), Source: Utility-Scale Solar, 2021 Edition
        'Wind':[30, Cost.iloc[0,2], 35.4]}                                          # Wind ($/MW), Source: Land-Based Wind Market Report: 2021 Edition
df_G = pd.DataFrame.from_dict(data)                                                 # save data into a dataframe
df_G.index = ['LS','Cost','CF']                                                     # add indices to the dataframe
df_rsr = pd.read_csv(os.path.join(future_data_dir,"dist_RSR_Nov-10 21-35_9.csv"))   # risk, size, REC
a0, b0 = [8, 8]                                                                     # initial alpha and beta parameters for risk distribution
risk_dist = np.array(df_rsr["risk"])                                                # agents are risk-averse
size_dist = np.array(df_rsr["size"])                                                # agents' sizes
rec_dist = np.array(df_rsr["REC"])                                                  # agents' perception about REC incentive (%) - smaller company recieves less financial incentive
df_cost = pd.read_csv(os.path.join(future_data_dir, "dist_cost_Nov-10 21-35_9.csv"))# cost perception for NG, Wind, and Solar
ng_cost_dist =  df_cost["NG"]                                                       # NG cost adjustment
solar_cost_dist = df_cost["Solar"]                                                  # solar cost adjustment
wind_cost_dist = df_cost["Wind"]                                                    # wind cost adjustment
bus_load = load_average(os.path.join(cwd,'ABM','Load_Data',
                                     'Loads_bus2gen_ERCOT'), 'ercot')               # get average loads for load profiles for each bus

# parameters
n_agt = 161                                                                         # number of agents; 74/161 are wind/solar, 8 solar only which makes 82/161 ~= 50%
renewable = int(0.5*n_agt)                                                          # number of renewalbe companies 
REC_p = 15                                                                          # renewable energy credit price ($/MW)
LZ = ['LZ_AEN','LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST']               # load zones
d_percent = [0.04,0.06,0.27,0.33,0.13,0.12]                                         # percentage to total demand based on 2020 data (=95% total demand)
D_zone_percent = pd.DataFrame(d_percent, index = LZ, columns = ['D_perc'])          # demand percentage dataframe
IRR_threshold = 0.08                                                                # IRR threshold (decision criteria)                                               
cap_initial = 128947                                                                # 2020 ERCOT total capacity (MW)                                                                                   

#%% future simulation
# df_risk = update_risk((temp_data_file, n_agt, risk_dist, a0, b0))                   # update risk values based on observed temperature
T = 30                                                                              # investment horizon starting from 2021
ys = 1                                                                              # year interval for investment (default value is 1)
n = 5                                                                               # investment period
load_scs = [f.path for f in os.scandir(dst) if f.is_dir()]                          # get directories of load scenarios
for load_sc in load_scs:                                                            # loop over load scenarios
    ### get necessary directories to read and save data
    pgms_frame_dir = os.path.join(load_sc, 'frame')                                 # PGMS frame directory (contains data needed to create input files for pyomo)
    gen_list_dir = os.path.join(load_sc, 'Generators')                              # generator list directory (single)
    gen_list_c_dir = os.path.join(load_sc, 'Generators_C')                          # generator list directory (commulative)
    pgms_loads_base_dir = os.path.join(load_sc, 'Loads_2Base')                      # base loads as a vector with hour 0 included
    pgms_loads_updated_dir = os.path.join(load_sc, 'Loads_3Updated')                # hourly updated loads after subtracting total renewable capacity
    pgms_inputs_dir = os.path.join(load_sc, 'PGMS_Inputs')                          # PGMS inputs directory
    pgms_outputs_dir = os.path.join(load_sc, 'PGMS_Outputs')                        # PGMS outputs directory
    
    ### initial values for variables
    tot_cap = cap_initial                                                           # initialize total capacity
    tot_invest = []                                                                 # create a list to store investment decisions
    row_list = []                                                                   # create a list to store investment decision for each year
    tot_new_cap = []                                                                # total new capacity installed each year 
    IRR_frame = []                                                                  # create a list to store mean IRR values   
    df_cap_renewable = pd.DataFrame()                                               # create an empty dataframe to store renewable capacity for each year at bus level
    bus_caps = []                                                                   # initial values
    linear = []                                                                     # supply curves [slope, intercept] of the load zones
    
    for t in range(T):                                                              # loop over years
        frame = []                                                                  # create a list to store investment decisions for each year
        ### PGMS
        y = 2021 + t                                                                # year
        # risk_dist = np.array(df_risk[y])                                          # select risk values for year y
        gen_list_cum = pd.read_csv(os.path.join(gen_list_c_dir, 
                                                os.listdir(gen_list_c_dir)[-1]), 
                                   index_col=False)                                 # read the last file
        base2input(y, bus_caps, df_cap_renewable, pgms_loads_base_dir, 
                   pgms_loads_updated_dir, gen_list_cum, pgms_frame_dir, 
                   pgms_inputs_dir)                                                 # create input files for PGMS from base load files
        pgms(pgms_inputs_dir, pgms_loads_updated_dir, gen_list_c_dir, 
              pgms_outputs_dir, y)                                                  # run PGMS model to determine price at each bus
        linear = pgms2abm(pgms_outputs_dir, y, bus_map, LZ, linear)                 # calculate supply curves [slope, intercept] of the load zones (price = a*demand + b)
        
        ### ABM
        agt_rec = REC_p*rec_dist                                                    # financial incentive for each agent
        annual_load = Annual_Demand.loc[y]['Energy (MWh)']                          # annual total load for year y
        new_cost = Cost.loc[y]                                                      # cost estimate for year y 
        cap_predicted = annual_load*0.00032 + 5638                                  # capacity prediction using demand forecast data (actual historical demand)
        cap_deficit = cap_predicted - tot_cap + Retire.loc[y][0]                    # capacity_deficit = predicted_capacity - current_capacity + retired_capacity
        if cap_deficit < 0:                                                         # if no demand for capacity, no investment
            cap_deficit = 0
        cap_deficit_agt = cap_deficit*size_dist                                     # capacity deficit for each agent
        new_capacity = 0                                                            # initial new investment capacity by each agent
        df_G.loc['CF'] = CF.loc[y]                                                  # update CF values for year y

        for agt_i in range(n_agt):                                                  # loop over agents
            IRR_t = []                                                              # create a list to store IRR values
            G_tech_lz = []                                                          # create a list to store technology types
            tech_row = {'Year':y, 'Agent':str(agt_i)}                               # a row that records investment technology for each load zone
            new_cost_dict = {"NG":new_cost["NG"]*ng_cost_dist[agt_i],               # create a dictionary to store cost values for each technology 
                             "Wind":new_cost["Wind"]*wind_cost_dist[agt_i], 
                             "Solar":new_cost["Solar"]*solar_cost_dist[agt_i]}
            df_G.loc['Cost'] = new_cost_dict                                        # update cost
    
            for i in range(len(LZ)):                                                # loop over loadzones
                c1,c2 = linear[i]                                                   # slope and intercept
                c3 = 0                                                              # scalar for agent's prediction error   
                """predicted average price for a load zone; annual_load is 
                divided by 365 because annual_load is annual but the price 
                prediction from the supply curve is based on daily data"""
                new_P = c1*annual_load/365*D_zone_percent.loc[LZ[i]].values[0] + c2
                rec_p = agt_rec[agt_i]                                              # REC value for the selected agent
                G_tech, max_IRR = ZoneInvest(new_P, rec_p, df_G)                    # determine the investment technology and the corresponding IRR
                # add investment decision results to new_row
                new_row = {'Year':y,'Agent':str(agt_i),'LZ':LZ[i],'Tech':G_tech,    # complete data for new row (investment by each agent)
                           'IRR':max_IRR} 
                tech_row[LZ[i]+"_tech"] = G_tech                                    # add investment technology in LZ to tech_row dictionary
                IRR_t.append(max_IRR)                                               # append the generated IRR to the IRR list
                row_list.append(new_row)                                            # append updated new_row to row_list
            
            IRR_t_array = np.array(IRR_t)                                           # convert IRR_t list to array
            agt_capacity_invest = cap_deficit_agt[agt_i]*risk_dist[agt_i]           # investment amount is discounted because of risk aversion. IOW,
                                                                                    # the agent will invest less than the needed capacity (hesitation).
            lz_invest = AgentInvest(LZ, IRR_t_array, agt_capacity_invest, 
                                    IRR_threshold)                                  # determine agent's investment in each load zone
            lz_invest_agt_sum = np.sum([*lz_invest.values()])                       # sum the capacity invested in load zones by the agent
            new_capacity += lz_invest_agt_sum                                       # add the agent's invested capacity to the total new capacity
            lz_invest.update(tech_row)                                              # merge the two dictionaries into "lz_invest"
            frame.append(lz_invest)                                                 # append investment by agent for each load zone to frame list
            IRR_frame.append(np.mean(IRR_t_array))                                  # append average IRR for all load zones for each agent
        
        tot_invest += frame                                                         # add investment decisions for each year to total investments
        tot_cap = tot_cap + new_capacity - Retire.loc[y][0]                         # add new capacity and subtract retirement
        tot_new_cap.append([y, cap_deficit, tot_cap])                               # append new installed capacity, capacity deficit, and the year to total new capacity
        
        # save results
        result_file = os.path.join(load_sc, "Agent_Investments", 
                                   "Agent_Investment_" + str(y) + ".csv")           # create a file name for result
        result_abm = pd.DataFrame(frame)                                            # convert list to dataframe
        result_abm.to_csv(result_file, index=False)                                 # save investment by agent for each load zone
        
        bus_caps = bus2gen(result_abm, bus_map, bus_load, gen_list_dir, y, ys)      # convert results at bus level to generator level
        gen_list_cum = gen_accumulation(gen_list_dir, gen_list_c_dir, y, ys, n)     # calculate cummulative generators list
        bus_caps.to_csv(os.path.join(load_sc, "Bus_Investments", "Bus_Investment_" 
                                     + str(y)+ ".csv"), index=False)                # save bus investments to *.csv file
        df_cap_renewable = cal_renewable(df_cap_renewable, y, bus_caps, df_G)       # calculate renewable capacity for year y
        
    # save total investment deicisons
    tot_result = pd.DataFrame(tot_invest)                                           # convert list to dataframe
    tot_result_file = os.path.join(load_sc, "Total_Agent_Investment.csv")           # create full file path to save results
    tot_result.to_csv(tot_result_file, index=False)                                 # save results to an excel file
    tot_annual_invest = pd.DataFrame(
        tot_new_cap, columns=["Year", "Capacity Deficit", "Total Capacity"])        # convert list to datafram    
    tot_annual_invest_file = os.path.join(load_sc, "Total_Annual_Investment.csv")   # create full file path to save total annual investments
    tot_annual_invest.to_csv(tot_annual_invest_file, index=False)                   # save results to an excel file
    df_cap_renewable_file = os.path.join(load_sc, "Renewable_Capacity.csv")         # create full file path to save annual renewable capacity
    df_cap_renewable.to_csv(df_cap_renewable_file, index=False)                     # save results to an excel file
    
    agg_tech, agg_year, agg_year_retire, agg_LZ = aggregate(n_agt, tot_result, 
                                                            Retire, cap_initial)    # aggregate investment decisions by technology, year, and load zone
    agg_tech_file = os.path.join(load_sc, "Technology by Agent.csv")                # create save file name for agg_tech                          
    agg_tech.to_csv(agg_tech_file, index=False)                                     # agg_tech: investment by technology types
    agg_year_file = os.path.join(load_sc, "Technology by Year.csv")                 # create save file name for agg_year
    agg_year.to_csv(agg_year_file)                                                  # agg_year: investment by year
    agg_year_retire_file = os.path.join(load_sc, "Technology by Year + Retirement.csv")   # create save file name for agg_year
    agg_year_retire.to_csv(agg_year_retire_file)                                    # agg_year: investment by year
    agg_LZ_file = os.path.join(load_sc, "Load Zone by Year.csv")                    # create save file name for agg_LZ
    agg_LZ.to_csv(agg_LZ_file)                                                      # agg_LZ: investment by load zone

elapsed_time = time.time() - st                                                             # measure elapsed time
print('Simulation is complete for {} years and {} scenarios.'.format(T, len(load_scs)))
print('Execution time:', datetime.datetime.now()-today)                                     # print a message to show elapsed time
