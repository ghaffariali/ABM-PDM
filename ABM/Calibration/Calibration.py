# -*- coding: utf-8 -*-
"""
Created on Fri Jul 05 16:49:06 2022

@author: Ali Ghaffari       alg721@lehigh.edu

Agent-based model 
If IRR>0.08 then the agent will invest, otherwise it won't. Investment amount
is determined by a normal distribution of the existing capacity. Based on data, 
a typical solar system in the U.S. can produce electricity at the cost of $0.06 
to $0.08 per kilowatt-hour. With a payback period of 5.5 years in New Jersey, 
Massachusetts, and California, the ROI is 18.2% annually. In some other states, 
the payback period can be up to 12 years, which translates to an 8.5% annual return.
"""

### Calibration
# import libraries
import sys
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from joblib import Parallel, delayed
sys.path.append(r'C:\Projects\ABM_PGMS\ABM')                                        # add to path
# import functions
from ABM_PGMS import read_data, supply_curve, generate_samples, ZoneInvest, \
    AgentInvest, aggregate, KGE_stat

#%% date
today = datetime.datetime.now()                                                     # create date object
date_now = today.strftime("%b-%d %H-%M")                                            # save date object as string
print("Today's date:", today)                                                       # output date

#%% read data files
# directories
os.chdir(r'C:\Projects\ABM_PGMS\ABM')                                               # change working directory
data_dir = r'C:\Projects\ABM_PGMS\ABM\Calibration\input'                            # data directory
result_dir = r'C:\Projects\ABM_PGMS\Results\00_Calibration'                         # results directory
demand_monthly = pd.read_csv(os.path.join(data_dir, 'demand_monthly.csv'))          # monthly demand data
price_monthly = pd.read_csv(os.path.join(data_dir, 'price_monthly.csv'))            # monthly price data
Demand = read_data(os.path.join(data_dir, 'demand_total.csv'))                      # read annual total demand (MWh) [Year,Demand]
Demand = Demand.set_index('Year')                                                   # set 'Year' columns as index
CF = read_data(os.path.join(data_dir, 'historical_capacity_factor_ABM.csv'))        # read capacity factor (%) [Year, NG, Wind, Solar, Battery] 
CP = read_data(os.path.join(data_dir, 'new_capacity_edited.csv'))                   # read new capacity (MW) by technology type [Year, NG, Solar, Wind, Total]
CA = read_data(os.path.join(data_dir, 'Retirement_Total_capacity.csv'))             # read retired, new, and total capacity (MW) [Year, New, Retired, Total]
CA = CA.set_index('Year')                                                           # set 'Year' column as index
G_hist_cost = read_data(os.path.join(data_dir, 'historical_cost_data.csv'))         # read total cost ($/MW) [Year, NG, Solar, Wind]
Retire = pd.read_csv(os.path.join(data_dir, 'Retire_hist.csv'), index_col='Year')   # historical retired capacity

#%% Calibration
"""in this section, random data is generated by gamma and normal distributions
to save size, risk, REC, and cost perceived by different agent types. For
each set of parameters, the code runs for 10 simluations to capture the
stochasticity of the model. Finally, the model with the best of set of
parameters will be selected to simulate future scenarios."""

dist_params = [0.8, 0.4, 0.35, 0.03, 0.7, 0.5, 8, 8]                                # distribution parameters = [mu_ng, sd_ng, mu_solar, sd_solar, mu_wind, sd_wind, risk_alpha, risk_beta]
K = 10                                                                              # number of simulations

def calibrate(k, dist_params, CF, CP, CA, Retire, G_hist_cost, Demand):
    """create a dict to store lifespan, cost, and capacity factor for all technologies
    lifespan based on Ziegler et. al, 2018; capacity factor based on 2019 data"""
    data = {'NG':[30, 1.510*1E6*1.1, 56.6],                                         # $917 in 2012 = $1510 in 2020 (5% interest rate); 10% O&M costs                                                             
            'Solar':[25, G_hist_cost['Solar'][0], 24.9],                            # solar price in 2000
            'Wind':[30, G_hist_cost['Wind'][0], 35.4]}                              # wind price in 2000
    df_G = pd.DataFrame.from_dict(data)                                             # save data into a dataframe
    df_G.index = ['LS','Cost','CF']                                                 # add indices to the dataframe

    # parameters
    n_agt = 161                                                                     # number of agents; 74/161 are wind/solar, 8 solar only, which makes 82/161 ~= 50%
    renewable = int(0.5*n_agt)                                                      # number of renewalbe companies 
    REC_p = 12                                                                      # Renewable Energy Credit price ($/MW)
    LZ = ['LZ_AEN','LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST']           # load zones 
    d_percent = [0.04,0.06,0.27,0.33,0.13,0.12]                                     # percentage to total demand based on 2020 data
    D_zone_percent = pd.DataFrame(d_percent, index = LZ, columns = ['D_perc'])      # demand percentage dataframe
    IRR_threshold = 0.08                                                            # IRR threshold (decision criteria) 
    
    # read historical data for comparison at the end   
    cap_hist = CP.iloc[12:,:]                                                       # historical new and total capacity 
    cap_retired = CA.iloc[13:,1:]                                                   # the capacity retired from 2012 - 2020 and the total capacity
    dist_cost, dist_RSR = generate_samples(dist_params, result_dir, renewable, 
                                           n_agt,date_now,k)                        # generate two dictionaries containing required random
    ng_cost_dist = dist_cost["NG"]                                                  # NG cost
    solar_cost_dist = dist_cost["Solar"]                                            # solar cost
    wind_cost_dist = dist_cost["Wind"]                                              # wind cost
    risk_dist = dist_RSR["risk"]                                                    # agents' risk perception distribution
    size_dist = dist_RSR["size"]                                                    # agents' size distribution
    rec_dist = dist_RSR["REC"]                                                      # agents' REC distribution

    # historical investment simulation
    agt_rec = REC_p*rec_dist                                                        # financial incentive for each agent
    frame = []                                                                      # create a list to store investment decisions
    row_list = []                                                                   # create a list to store investment decisions for each year
    tot_ca = CA.loc[2011][2]                                                        # total planned generation capacity in 2011 = 109179 MW
    tot_new_ca = []                                                                 # new total capacity installed each year

    for t in range(9):                                                              # zonal demand data only available from 2011 to 2020
        y = 2012 + t                                                                # year
        new_D = Demand.loc[y].values                                                # annual total demand for year y
        new_cost = G_hist_cost.iloc[:,1:][G_hist_cost["Year($MW)"] == y]            # remove column 'Year' and select the cost data for year y
        cap_predicted = new_D*0.00029 + 5638                                        # capacity prediction using demand forecast data (actual historical demand)
        cap_deficit = cap_predicted - tot_ca + CA.loc[y][1]                         # capacity_deficit = predicted_cpacity - current_capacity + retired_capacity

        if cap_deficit < 0:                                                         # if no demand for capacity, no investment
            cap_deficit = 0

        linear = supply_curve(LZ, y)                                                # supply curves [slope, intercept] of the load zones
        cap_deficit_agt = cap_deficit*size_dist                                     # capacity deficit for each agent
        new_capacity = 0                                                            # initial new capacity investment by each agent
        for agt_i in range(n_agt):                                                  # loop over agents
            IRR_t = []                                                              # create a list to store IRR values
            G_tech_lz = []                                                          # create a list to store technology types
            tech_row = {'Year':y, 'Agent':str(agt_i)}                               # a row that records investment technology for each load zone
            new_cost_dict = {"NG":new_cost["NG"].values[0]*ng_cost_dist[agt_i],     # create a dictionary to store cost values for each technology
                             "Solar":new_cost["Solar"].values[0]*solar_cost_dist[agt_i],
                             "Wind":new_cost["Wind"].values[0]*wind_cost_dist[agt_i]}
            df_G.loc['Cost'] = new_cost_dict                                        # update cost
    
            for i in range(len(LZ)):                                                # loop over load zones
                c1,c2 = linear[i]                                                   # slope and intercept (price = a*demand + b)
                c3 = 0                                                              # scalar for agent's prediction error   
                """predicted average price for a load zone; new_d is divided by
                12 because new_d is annual but the price prediction from the
                supply curve is based on monthly data"""
                new_P = c1*new_D/12*D_zone_percent.loc[LZ[i]].values[0] + c2        # calculate new price
                rec_p = agt_rec[agt_i]                                              # REC value for the selected agent
                G_tech, max_IRR = ZoneInvest(new_P, rec_p, df_G)                    # determine the investment technology and the corresponding IRR
                new_row = {'Year':y, 'Agent':str(agt_i), 'LZ':LZ[i],                
                           'Tech':G_tech, 'IRR':max_IRR}                            # add investment decision results to new_row
                tech_row[LZ[i]+"_fuel"] = G_tech                                    # add the investment technology in the LZ to tech_row dictionary
                IRR_t.append(max_IRR)                                               # append the generated IRR to the IRR list
                row_list.append(new_row)                                            # append updated new_row to row_list
            IRR_t_array = np.array(IRR_t)                                           # convert IRR_t list to array
            """investment amount is discounted because of risk aversion. IOW,
            the agent will invest less than the needed capacity (hesitation)."""
            agt_capacity_invest = cap_deficit_agt[agt_i]*risk_dist[agt_i]
            lz_invest = AgentInvest(LZ, IRR_t_array, agt_capacity_invest,           
                                    IRR_threshold)                                  # determine agent's investment in each load zone
            lz_invest_agt_sum = np.sum([*lz_invest.values()])                       # sum the capacity invested in load zones by the agent
            new_capacity += lz_invest_agt_sum                                       # add the agent's invested capacity to the total new capacity
            lz_invest.update(tech_row)                                              # update lz_invest dict
            frame.append(lz_invest)                                                 # append investment by load zones to frame list
        
        tot_ca = tot_ca + new_capacity - cap_retired.loc[y][0]                      # add new capacity and subtract retirement
        tot_new_ca.append(tot_ca)                                                   # append new installed capacity to total new capacity
    
    result = pd.DataFrame(frame)                                                    # save frame to a dataframe
    cap_initial = CP.iloc[11,4]                                                     # initial capacity
    agg_tech, agg_year, agg_LZ = aggregate(n_agt, result, Retire, cap_initial)      # aggregate investment decisions by technology, year, and load zone
    agg_year_file = os.path.join(result_dir, date_now, "agg_year" + str(k) +".csv")
    agg_year.to_csv(agg_year_file)                                                  # agg_year: investment by year
    tech_cumulative = agg_year.sum()                                                # cumulative investment in MW at the end of 2020
    
    kge, p = KGE_stat(agg_year, tot_new_ca, cap_hist)                               # calculate KGE
    kge['Wind_cum'] = tech_cumulative['Wind']
    kge['Solar_cum'] = tech_cumulative['Solar']
    kge['NG_cum'] = tech_cumulative['NG']
    kge['Total_cum'] = tech_cumulative['Total']
    kge['Simulation'] = k
    df_KGE = pd.DataFrame([kge])                                                    # save KGE results to a dataframe
    df_KGE.to_csv(os.path.join(result_dir, date_now,'kge_{}.csv'.format(k)), 
                  index=False)                                                      # save to *.csv
    
    ### plot
    time = np.arange(2012, 2021)                                                    # time for x axis in plot
    df = CP.iloc[-9:,:]                                                             # extract data for 2012 to 2021
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(time, df['NG'], color='b')     
    axs[0, 0].plot(time, agg_year['NG'], color='y')
    axs[0, 0].set_title('NG')
    axs[0, 1].plot(time, df['Solar'], color='b')
    axs[0, 1].plot(time, agg_year['Solar'], color='y')     
    axs[0, 1].set_title('Solar')
    axs[1, 0].plot(time, df['Wind'], color='b')
    axs[1, 0].plot(time, agg_year['Wind'], color='y')  
    axs[1, 0].set_title('Wind')
    axs[1, 1].plot(time, df['Total'], color='b')
    axs[1, 1].plot(time, agg_year['Total_cum'], color='y')   
    axs[1, 1].set_title('Total')
    
    for ax in axs.flat:                                                                 # set xtick labels
        ax.set_xticks(np.arange(2012,2021,2))
        ax.set_xticklabels(np.arange(2012,2021,2))
    
    # set x and y labels
    axs[0,0].set_title('NG')
    axs[0,0].set_ylabel('Capacity (MW)')                                        
    axs[0,1].set_title('Solar')    
    axs[1,0].set_title('Wind')
    axs[1,0].set_xlabel('Year') 
    axs[1,0].set_ylabel('Capacity (MW)') 
    axs[1,1].set_title('Total')
    axs[1,1].set_xlabel('Year')

    # create label for colors
    b_patch = mpatches.Patch(color='b', label='Historical')
    y_patch = mpatches.Patch(color='y', label='Calibrated')
    axs[0,1].legend(handles=[b_patch, y_patch], loc=2, fontsize='xx-small')         # only put legend for axs[0,1]
                            
    fig.suptitle('Calibration results for simulation {}'.format(k))
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, date_now, 'Simulation' +str(k)+'.png'), 
                dpi=300)        

Parallel(n_jobs=-1)\
    (delayed(calibrate)\
     (k, dist_params, CF, CP, CA, Retire, G_hist_cost, Demand) for k in range(K))

frame = []
for i in range(K):
    kge_file = os.path.join(result_dir, date_now, 'kge_{}.csv'.format(i))           # KGE file for simulation i
    kge = pd.read_csv(kge_file)                                                     # read csv file
    frame.append(kge)                                                               # append df to frame
    os.remove(kge_file)                                                             # remove KGE file

df_KGE = pd.concat(frame)                                                           # concat frame to get one KGE matrix for all simulations
ng_cap = CP.iloc[-10:,1].sum()                                                      # calculate cumulative solar capacity
solar_cap = CP.iloc[-10:,2].sum()                                                   # calculate cumulative solar capacity
wind_cap = CP.iloc[-10:,3].sum()                                                    # calculate cumulative wind capacity
total_cap = ng_cap + solar_cap + wind_cap
df_KGE['NG_fit'] = 1-abs(df_KGE['NG_cum']-ng_cap)/df_KGE['NG_cum']
df_KGE['Solar_fit'] = 1-abs(df_KGE['Solar_cum']-solar_cap)/df_KGE['Solar_cum']
df_KGE['Wind_fit'] = 1-abs(df_KGE['Wind_cum']-wind_cap)/df_KGE['Wind_cum']
# df_KGE['Total_fit'] = 1-abs(df_KGE['Total_cum']-total_cap)/df_KGE['Total_cum']

df_KGE['Total_fit'] = (2*df_KGE['NG_fit']+4*df_KGE['Solar_fit']+4*df_KGE['Wind_fit'])/10
df_KGE_file = os.path.join(result_dir, date_now, "KGE.csv")
df_KGE.to_csv(df_KGE_file, index=False)                                             # save KGE results to *.csv

with open(os.path.join(result_dir, date_now, 'dist_params.txt'), 'w') as fp:        # save dist_params
    for item in dist_params:
        # write each item on a new line
        fp.write("%s\n" % item)
