# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:50:37 2022

@author: Ali Ghaffari       alg721@lehigh.edu

this script contains functions & classes needed to run coupled ABM-PGMS model.
"""
#%% initialization
# import libraries
import os
import numpy as np
import pandas as pd
import pickle
import scipy.stats
import numpy_financial as npf                                                           # import npf to calculate IRR (Internal Rate of Return)
import shutil, errno                                                                    # import shutil to copy source folder
import scipy.stats as stats
pd.options.mode.chained_assignment = None                                               # turn off warnings (default='warn')
from sklearn.linear_model import LinearRegression
from scipy import interpolate

### Classes
# =============================================================================
# generator
# =============================================================================
class generator():
    def __init__(self, name, cost, ls):
        self.name = name        # name of the generator
        self.cost = cost        # the invest. amount
        self.life_span = ls     # the life span of the generator


### Functions
# =============================================================================
# copydirectory
# =============================================================================
"""this function copies whole directory"""
def copydirectory(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise

# =============================================================================
# pgms2abm
# =============================================================================
"""this function converts PGMS output to ABM input; PGMS yields price, load, 
and power output for each bus for n load scenarios. this data needs to be 
converted to n pairs of [load, price] to predict future price in ABM."""
def pgms2abm(input_dir, y, bus_map, LZ, linear):
    try:
        input_files = os.listdir(os.path.join(input_dir, str(y), '02_Prices'))                           # list output files by PGMS model for year y
        linear = []                                                                         # create a list to store regression results
        df_load = pd.DataFrame()                                                            # create a dataframe to store load by loadzone results
        df_load["Bus_Number"] = np.arange(1,124,1)                                          # add a column as bus number
        df_load["LZ"] = bus_map["LZ"]                                                       # add a column as bus loadzone
        df_price = pd.DataFrame()                                                           # create a dataframe to store price by loadzone results
        df_price["Bus_Number"] = np.arange(1,124,1)                                         # add a column as bus number
        df_price["LZ"] = bus_map["LZ"]                                                      # add a column as bus loadzone
        for i in range(len(input_files)):                                                   # loop over PGMS outputs for load scenarios
            input_file = input_files[i]                                                     # ith input file
            df = pd.read_csv(os.path.join(input_dir, str(y), '02_Prices', input_file))                   # read input file as *.csv file
            df_load["Load"+str(i+1)] = df["Load"]                                           # add a column as Load_i to load dataframe
            df_price["Price"+str(i+1)] = df["Price"]                                        # add a column as Price_i to price dataframe
        for lz in LZ:                                                                       # loop over loadzones
            subset_load = df_load.loc[df_load["LZ"] == lz]                                  # select only buses within the selected load zone                                       
            load = subset_load.iloc[:,2:].sum(axis=0).values                                # calculate load for each loadzone for each scenario
            X = load.reshape(-1,1)                                                          # reshape data for regression
            subset_price = df_price.loc[df_price["LZ"] == lz]                               # select only buses within the selected load zone
            price = subset_price.iloc[:,2:].mean(axis=0).values                             # calculate price for each loadzone for each scenario
            reg = LinearRegression().fit(X, price)                                          # create a linear regression model as price = a*load + b
            linear.append([reg.coef_[0],reg.intercept_])                                    # append [slope, intercept] of the linear supply curve for each load zone
    except:                                                                                 # if PGMS_Outputs/y does not exist
        print('No solution found for year {}.'.format(y))
        print('ABM will use price values for year {}'.format(y-1))        
    
    return linear                                                                           # return a list of [slope, intercept] values for 16 scenarios for 6 load zones   

# =============================================================================
# agt2bus
# =============================================================================
"""this function converts the results into proper format for the energy model.
ABM results are in agent level which could also be aggregated to load zone 
level. Each bus could be attributed to one load zone and therefore, the results 
are converted to bus level using weighted average based on load at each bus. 
The final output consists of new capacity for each bus based on different 
technology types (i.e. NG, Solar, and Wind)."""
def agt2bus(result_abm, bus_map, bus_load, LZ, techs):
    bus_map['Load'] = bus_load['Average Load']                                          # add previous capacity as a column to bus_map
    frame1, frame2 = [], []                                                             # create a list to store results for each load zone and bus
    for lz in LZ:                                                                       # loop over load zones
        # first, load zone capacity is determined by technology type
        subset_tech = result_abm[[lz, lz+'_tech']]                                      # subset data by year, load zone, and technology                
        dummy = subset_tech.groupby(lz+'_tech').sum()                                   # group data by technology and sum
        frame1.append(dummy)                                                            # append load zone capacity by technology to frame
        # second, load zone capacity is converted to bus capacity.
        subset_lz = bus_map.loc[bus_map['LZ'] == lz]                                    # select only buses within the selected load zone
        subset_lz['bus_coeff'] = subset_lz['Load']/sum(subset_lz['Load'])               # calculate generation coefficient as a weighted average
        for tech in techs:                                                              # loop over technologies
            if tech in dummy.index:                                                     # if the technology is in the list of investments
                subset_lz[tech] = subset_lz['bus_coeff']*dummy[lz][tech]                # calculate the new capacity at each bus
            else:                                                                       # if the technology is not in the list of investments
                subset_lz[tech] = 0                                                     # new capacity at bus is 0
        frame2.append(subset_lz)                                                        # append new capacity to frame2
    lz_caps = pd.concat(frame1, axis=1)                                                 # store load zone capacities in one dataframe
    bus_caps = pd.concat(frame2, ignore_index=True)                                     # store bus capacities in one dataframe
    bus_caps = bus_caps.loc[:, ['Bus_Number','NG','Solar','Wind']]                      # clean the data
    bus_caps.sort_values('Bus_Number', inplace=True, ignore_index=True)                 # sort the data by bus number 
    
    return lz_caps, bus_caps                                                            # return load zone and bus capacities

# =============================================================================
# bus2gen
# =============================================================================
"""this function converts investment decisions at bus level to generator level; 
ABM yields investment amount and technology for each agent. this data needs 
to be converted to bus level (done by agt2bus function). then, for each bus, 
three generator types as NG, Solar, and Wind will be added to generators data. 
there are other parameters that need to be determined for generators. this 
part is technical and here, I have used average values."""
def bus2gen(result_abm, bus_map, bus_load, gen_list_dir, y, ys):
    gen_list_raw = pd.read_csv(os.path.join(gen_list_dir, "gen_list_" + str(y) + ".csv"))
    del gen_list_raw["Index"]                                                           # delete "Index" column
    LZ = ['LZ_AEN','LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST']               # load zones
    techs = ['NG', 'Solar', 'Wind']                                                     # technology types
    lz_caps, bus_caps = agt2bus(result_abm, bus_map, bus_load, LZ, techs)               # convert investment decisions at agent-level to bus-level
    # Source: https://www.lazard.com/perspective/levelized-cost-of-energy-levelized-cost-of-storage-and-levelized-cost-of-hydrogen/#:~:text=The%20former%20values%20average%20%2427,for%20combined%20cycle%20gas%20generation.
    # 27.9 comes from a weighted average between coal, natural gas, and nuclear
    gen_data = {'NG': [27.9, 2.41, 0.35, 0, 10.48],                                     # gen_cost_P, gen_cost_NL, gen_cost_SU, gen_Pmin, gen_rr
                'Solar': [27, 1, 0, 0, 0],
                'Wind': [25, 1, 0, 0, 0]}
    df_gen = pd.DataFrame.from_dict(gen_data)                                           # create a dataframe out of dictionary
    df_gen.index= ['gen_cost_P','gen_cost_NL','gen_cost_SU','gen_Pmin','gen_rr']        # set index for the dataframe
    for bus in bus_caps["Bus_Number"]:                                                  # loop over buses
        tech = "NG"                                                                     # only NG will be added as new generators to pyomo input file
        new_cap = bus_caps.loc[bus-1, tech]                                             # find new capacity at the bus for the technology type
        if new_cap > 0.01:                                                              # if new capacity for the technology larger than 0.01
            new_gen = {'gen_num': [int(max(gen_list_raw["gen_num"]))+1],                # generator's number
                       'gen_bus': [bus],                                                # generator's bus number☺
                       'gen_cost_P': [np.random.normal(loc=27.9, scale=5.0)],           # generator's operational cost ($/MWh) (random normal value)
                       'gen_cost_NL': [df_gen[tech]['gen_cost_NL']*new_cap],            # generator's no load cost ($/hr)
                       'gen_cost_SU': [df_gen[tech]['gen_cost_SU']*new_cap],            # generator's start-up cost ($/start-up)
                       'gen_Pmin': [df_gen[tech]['gen_Pmin']*new_cap],                  # generator's minimum power when generator is online (MW)
                       'gen_Pmax': [new_cap],                                           # generator's maximum power when generator is online (MW)
                       'gen_r10': [df_gen[tech]['gen_rr']*new_cap/6],                   # generator's ramping rate in 10 minutes (MW/10min)
                       'gen_rr': [df_gen[tech]['gen_rr']*new_cap],                      # generator's ramping rate (MW/hr)
                       'gen_tech': [tech],                                              # generator's technology type
                       'gen_year': y+ys}                                                # generator's construction year                                              
            df_new_gen = pd.DataFrame(new_gen)                                          # convert dictionary to dataframe
            gen_list_raw = pd.concat([gen_list_raw,df_new_gen], ignore_index=True)      # add new generator data to generator's list
    gen_list_updated = gen_list_raw                                                     # update generators list
    gen_list_updated.insert(0, "Index", np.arange(1,len(gen_list_raw)+1,1))             # add a column as index
    gen_list_updated.to_csv(os.path.join(gen_list_dir, "gen_list_" + str(y+ys) 
                                         + ".csv"), index=False)                        # save dataframe to *.csv file
    
    return bus_caps                                                                     # return bus_caps

# =============================================================================
# accu_gens
# =============================================================================
"""this function accumulates annual generators list into one list by summation
of new generators based on their bus. This is done to speed up the PGMS code. 
Also it is more realistic to think that new projects are completed in n-year
cycles rather than being built separately for each year. IOW, the new 
generators are combined into one generator for each bus every n years."""
def gen_accumulation(gen_list_dir, gen_list_c_dir, y, ys, n, base_year=2021):
    """this little function identifies years who belong to one investment period"""
    def id_invest_pd(gen_list_dir, n, base_year=2021):
        years = [int(X[-8:-4]) for X in os.listdir(gen_list_dir)][::-1]                 # get years and reverse the list
        years_pd = []                                                                   # create a list to store years in one investment period
        for i in years[:n]:                                                             # loop over the last n years
            if (i-base_year)%n != 0:                                                    # y belongs to the investment period
                years_pd.append(i)
            elif (i-base_year)%n == 0 and i == max(years):                              # y is the last year in the investment period
                years_pd.append(i)
            elif (i-base_year)%n == 0 and i != max(years):                              # y is the last year in the previous investment period
                break
        return years_pd[::-1]                                                           # return years in one investment period
    
    try:
        ### read all generator list data for one investment period
        years_pd = id_invest_pd(gen_list_dir, n, base_year=2021)
        frame = []                                                                      # create a list to store dfs
        for year in years_pd:                                                           # loop over years in the investment period
            df = pd.read_csv(os.path.join(gen_list_dir, 'gen_list_' 
                                          + str(year) + '.csv'))                        # read generators list for year
            frame.append(df[df['gen_year'] == year])                                    # append df to frame
        df_period_n = pd.concat(frame, ignore_index=True)                               # concat all dfs from the same investment period
        ### filter by bus
        frame = []
        for bus in np.arange(1,124):                                                    # loop over 123 bus
            if bus in df_period_n['gen_bus'].values:                                    # if the bus has any generators at all
                df_bus = df_period_n[df_period_n['gen_bus'] == bus]                     # slice generators for one bus
                df_bus_c = df_bus.groupby('gen_bus').mean()                             # cummulative df_bus
                df_bus_c['gen_Pmax'] = df_bus.groupby('gen_bus').sum()['gen_Pmax']      # use sum of Pmax value for the generator
                df_bus_c.insert(9, "gen_tech", "NG")                                    # insert gen_type
                df_bus_c.insert(1, "gen_bus", df_bus.iloc[0]['gen_bus'])                # insert gen_bus
                frame.append(df_bus_c)
        df_bus_total = pd.concat(frame)                                                 # concat all bus_dfs
        df_bus_total['gen_year'] = y + ys                                               # update year
        df_base = pd.read_csv(os.path.join(gen_list_c_dir, 'gen_list_c_' + 
                              str(min(years_pd)-1) + '.csv'))                           # read cummulative generators list from last investment period
        gen_list_cum = pd.concat([df_base, df_bus_total], 
                                 ignore_index=True).round(decimals=2)                   # concat df_base and new cummulative df_bus
        gen_list_cum['Index'] = np.arange(1,len(gen_list_cum)+1)                        # correct Index values
        
    except ValueError:
        if len(frame) == 0:                                                             # if no new NG generators are added
            gen_list_cum = pd.read_csv(os.path.join(gen_list_c_dir, "gen_list_c_" 
                                                    + str(y) + ".csv"))                 # read gen_list_cum for previous year
            gen_list_cum['gen_year'] = np.where(gen_list_cum['gen_year'] == 
                                                gen_list_cum['gen_year'].max(), 
                                                y+ys, gen_list_cum['gen_year'])         # set gen_year to y+ys to update gen_list_cum
    gen_list_cum['gen_num'] = gen_list_cum['gen_num'].astype(int)                       # gen_num should be integer
    gen_list_cum.to_csv(os.path.join(gen_list_c_dir, "gen_list_c_" + str(y+ys) + 
                                 ".csv"), index=False)                                  # save gen_list_cum    
    return gen_list_cum                                                                 # return cummulative generators 

# =============================================================================
# initial_load
# =============================================================================
"""this function converts initial load (2021) to the correct format by adding 
initial load as 0 and reshaping the load file into one column."""
def initial_load(pgms_loads_raw_dir, load_gcam_dir):
    loads = os.listdir(pgms_loads_raw_dir)                                              # get a list of all files in load directory
    y = 2021                                                                            # initial year
    os.mkdir(os.path.join(load_gcam_dir, str(y)))                                       # create directory to store 2021 loads
    for load in loads:                                                                  # loop over loads
        output_file = os.path.join(load_gcam_dir, str(y), load)                         # create a filename for the output
        zero_load = pd.DataFrame(np.zeros([1,123]))                                     # create a row of zeros as initial load
        new_load = pd.read_csv(os.path.join(pgms_loads_raw_dir, load), 
                               delimiter="\t", header=None)                             # read load data for 24 hours
        total_load = pd.concat([zero_load, new_load], ignore_index=True)                # concatenate two dataframes
        total_load = total_load.transpose().values.reshape(-1,1)                        # reshape the dataframe to add as a column to frame
        df_load = pd.DataFrame(total_load)                                              # create a dataframe out of frame list
        df_load.to_csv(output_file, index=False, header=False)                          # save manipulated load to *.csv file

# =============================================================================
# load_average
# =============================================================================
"""this function reads 16 load scenarios and takes average to have load at each 
bus. this data is later used to convert results at bus level to generator
level."""
def load_average(pgms_loads_raw_dir, source):
    load_files = os.listdir(pgms_loads_raw_dir)                                         # get a list of all files in load directory
    frame = []                                                                          # create a list to store results
    for load in load_files:                                                             # loop over load files
        if source == 'gcam':
            new_load = pd.read_csv(os.path.join(pgms_loads_raw_dir, load), 
                                   delimiter="\t", header=None)                         # open ith load file
            new_load = new_load.mean(axis=0)                                            # calculate mean load to convert hourly data to daily data
        elif source == 'ercot':
            new_load = pd.read_csv(os.path.join(pgms_loads_raw_dir, load), 
                                   delimiter=" ", header=None)                          # open ith load file
            new_load = new_load.T.mean(axis=0)                                          # calculate mean load to convert hourly data to daily data
        frame.append(new_load)                                                          # append daily load for 123 buses to frame
    df = pd.DataFrame(frame)                                                            # create a dataframe out of list
    bus_load = pd.DataFrame()                                                           # create a dataframe to store average load for buses for 16 scenarios
    bus_load["Average Load"] = df.mean(axis=0)                                          # calculate average load for all load profiles
    
    return bus_load                                                                     # return average load for 123 buses

# =============================================================================
# update_load    
# =============================================================================
"""this function updates load data by deducing renewable capacity so that the 
load values contian only dispatchable load. Updated load values are stored in
'Loads_Updated'."""
# def cal_renewable(df_cap_renewable, y, bus_caps):
#     cap_solar = bus_caps["Solar"].values                                                # solar capacities
#     cap_wind = bus_caps["Wind"].values                                                  # wind capacities
#     load_frame = []; cap_renewable = []                                                 # create an empty list to store results
#     index = 1                                                                           # initial index value
#     # create a frame for hourly load at each bus
#     for i in range(1,124):                                                              # loop over bus numbers
#         for t in range(0,7):                                                            # 00:00 to 7:00 
#             load_frame.append([i, t, index])                                            # append a new row to frame
#             cap_renewable.append(cap_wind[i-1])                                         # only wind (no sunlight)
#             index += 1                                                                  # add index by 1
#         for t in range(7,19):                                                           # 7:00 to 19:00
#             load_frame.append([i, t, index])                                            # append a new row to frame
#             cap_renewable.append(cap_wind[i-1] + cap_solar[i-1])                        # solar and wind
#             index += 1                                                                  # add index by 1
#         for t in range(19,25):                                                          # 19:00 to 24:00
#             load_frame.append([i, t, index])                                            # append a new row to frame
#             cap_renewable.append(cap_wind[i-1])                                         # only wind (no sunlight)
#             index += 1                                                                  # add index by 1
#     if y==2021:                                                                         # initial year
#         df_cap_renewable = pd.DataFrame(load_frame, columns=['bus','hour','bus-hour'])  # put load_frame in df_cap_renewable
#     df_cap_renewable[y] = cap_renewable                                                 # save renewable capacity to column y of df_cap_renewable
    
#     return df_cap_renewable                                                             # return a dataframe containing renewable capacity at bus level for each year

def cal_renewable(df_cap_renewable, y, bus_caps, df_G):
    cap_solar = bus_caps["Solar"].values                                                # solar capacities
    cap_wind = bus_caps["Wind"].values                                                  # wind capacities
    load_frame = []; cap_renewable = []                                                 # create an empty list to store results
    index = 1                                                                           # initial index value
    # create a frame for hourly load at each bus
    for i in range(1,124):                                                              # loop over bus numbers
        for t in range(0,25):                                                           # loop over 24 hours 
            load_frame.append([i, t, index])                                            # append a new row to frame
            cap_renewable.append(cap_solar[i-1]*df_G['Solar']['CF']/100 + 
                                 cap_wind[i-1]*df_G['Wind']['CF']/100)                  # solar + wind
            index += 1                                                                  # add index by 1

    if y==2021:                                                                         # initial year
        df_cap_renewable = pd.DataFrame(load_frame, columns=['bus','hour','bus-hour'])  # put load_frame in df_cap_renewable
    df_cap_renewable[y] = cap_renewable                                                 # save renewable capacity to column y of df_cap_renewable
    
    return df_cap_renewable                                                             # return a dataframe containing renewable capacity at bus level for each year


# def cal_renewable(df_cap_renewable, y, bus_caps, df_G):
#     """this section determines hours at which the power unit operates at max
#     capacity. this part is to consider CF in load profiles."""
#     def apply_cf(df_G, tech, cap_tech):
#         hours = np.zeros(25)                                                            # create a list of zeros to determine hours with max capacity
#         max_time = np.round(df_G[tech]['CF']/100*12)                                    # determine time that power unit operates at max capacity
#         for h in range(25):                                                             # loop over hours
#             if h in np.arange(12-max_time, 12+max_time+1):                              # if h is in the time with max capacity
#                 hours[h] = 1                                                            # 1 means operating at full capacity; 0 otherwise
#         load_frame = []; cap_temp = []                                                  # create an empty list to store results
#         # create a frame for hourly load at each bus
#         index = 1
#         for i in range(1,124):                                                          # loop over bus numbers
#             for t in range(25):                                                         # loop over 24 hours
#                 load_frame.append([i, t, index])                                        # append a new row to frame
#                 if hours[t] == 1:                                                       # if power unit operates at max capacity
#                     cap_temp.append(cap_tech[i-1])                                      # capacity = nameplate_capacity
#                 else:                                                                   # if power unit does not operate at max capacity
#                     cap_temp.append(cap_tech[i-1]*np.random.rand()/5)                   # capacity = nameplate_capacity*random number/5
#                 index += 1
#         return load_frame, cap_temp

#     load_frame, cap_solar = apply_cf(df_G, 'Solar', bus_caps["Solar"].values)           # calculate solar capacity while applying CF
#     load_frame, cap_wind = apply_cf(df_G, 'Wind', bus_caps["Wind"].values)              # calculate wind capacity while applying CF
#     if y==2021:                                                                         # initial year
#         df_cap_renewable = pd.DataFrame(load_frame, columns=['bus','hour','bus-hour'])  # put load_frame in df_cap_renewable
#     df_cap_renewable[y] = [sum(x) for x in zip(cap_solar, cap_wind)]                    # save renewable capacity (solar+wind) to column y of df_cap_renewable
    
#     return df_cap_renewable                                                             # return a dataframe containing renewable capacity at bus level for each year

# =============================================================================
# base2input
# =============================================================================
"""this function creates input files for PGMS from base loads for both GCAM and
ERCOT loads by subtracting total renewable capacity to get dispatchable load. 
loads_base_dir: base loads as a vector with hour 0 included
loads_updated_dir: loads after subtracting total renewable capacity"""
def base2input(y, bus_caps, df_cap_renewable, loads_base_dir, loads_updated_dir, 
               gen_list_cum, frame_dir, pgms_inputs_dir):
    base_loads = os.listdir(os.path.join(loads_base_dir, str(y)))                       # read hourly load data from GCAM; this is the conversion basis
    os.mkdir(os.path.join(pgms_inputs_dir, str(y)))                                      # create a folder to store PGMS input files for year y
    index = 1                                                                           # initial index value
    # create a frame for hourly load at each bus    
    load_frame = []                                                                     # create a list to store frame data
    for i in range(1,124):                                                              # loop over bus numbers
        for t in range(0,25):                                                           # loop over 24 hours including hour 0
            load_frame.append([i, t, index])                                            # append a new row to frame
            index += 1                                                                  # add index by 1
    df_load = pd.DataFrame(load_frame)                                                  # create a dataframe out of frame list
    
    os.mkdir(os.path.join(loads_updated_dir, str(y)))                                   # create a folder to store updated loads
    for load in base_loads:                                                             # loop over load data
        base_load = pd.read_csv(os.path.join(loads_base_dir, str(y), load), header=None)# read hourly load data in one column
        if y==2021:                                                                     # initial load
            df_load['load'] = base_load[0]                                              # calculate new load after multiplying by annual ERCOT load and dividing by 365
            gen_list_raw = open(os.path.join(frame_dir, "gen_list_raw.txt"), 'r')       # Part2: generator list
        else:
            df_load['load'] = base_load[0] - df_cap_renewable.iloc[:,3:].sum(axis=1)    # update load by reducing renewable cap - first 3 columns are not load data
            df_load = df_load.round(2)                                                  # round load by 2 decimals
            df_load['load'][df_load['load'] < -0.00001] = 0                             # set negative load values to zero (if any)
            gen_list_raw = gen_list_cum.iloc[:, :-2]                                    # select all columns except technology type and gen_year (because it is a string)
        updated_load_file = os.path.join(loads_updated_dir, str(y), load)               # create file full path for updated load
        df_load['load'].to_csv(updated_load_file, index=False, header=False)            # save dataframe to *.txt file
        output_file = os.path.join(pgms_inputs_dir, str(y),                              # create file full path for pgms input file
                                   "formpyomo_UC_data_"+load[5:-3]+"dat")               # create output file name
        bus_list_raw = open(os.path.join(frame_dir, "bus_list_raw.txt"), 'r')           # Part1: bus list
        line_list_raw = open(os.path.join(frame_dir, "line_list_raw.txt"), 'r')         # Part3: line list
        time_list_raw = open(os.path.join(frame_dir, "time_list_raw.txt"), 'r')         # Part4: time list
        with open(output_file, 'a') as f:                                               # open output file as append
            # order: bus, generator, line, time, load
            f.write(bus_list_raw.read())                                                # Part1: bus list
            if y==2021:                                                                 # initial load
                f.write(gen_list_raw.read())                                            # Part2: generator list
            else:
                df_gen_string = gen_list_raw.to_string(header=False, index=False)       # convert dataframe to string
                f.write(df_gen_string)                                                  # Part2: generator list
            f.write(line_list_raw.read())                                               # Part3: line list
            f.write(time_list_raw.read())                                               # Part4: time list
            df_load_string = df_load.to_string(header=False, index=False)               # convert dataframe to string
            f.write(df_load_string)                                                     # Part5: bus-load list
            f.write('\n;')                                                              # add one ; at the end as a section-end marker
            f.close()                                                                   # close file

# =============================================================================
# update_risk
# =============================================================================
"""this function updates risk values based on observed temperature data. Agents'
climate risk perception increases if they observe higher temperatures in 
summer and lower temperatures in winter."""
def update_risk(temp_file, n_agt, initial_risk, a0, b0, dst):
    # =============================================================================
    # annual_data
    # =============================================================================
    """this function reads the data and outputs a dataframe where each column 
    represents data for one year (mode 'm' for monthly and mode 'd' for daily)."""
    def annual_data(data, mode):
        df_annual = pd.DataFrame()                                                      # create a dataframe to save annual data in each column
        data["DATE"] = pd.to_datetime(data["DATE"])                                     # set "DATE" column as datetime
        yr1 = data["DATE"].min().year                                                   # initial year
        yr2 = data["DATE"].max().year                                                   # final year
        if mode == 'd':
            for year in np.arange(yr1, yr2+1):                                          # loop over years
                annual_data = data['TAVG'].loc[(data["DATE"] >= str(year)+'-01-01')     # get average temperature data (TAVG) for one year
                               & (data["DATE"] <= str(year)+'-12-31')]
                df_annual[year] = annual_data[:365]
        elif mode == 'm':
            data = data.resample('MS', on='DATE').mean()                                # resample data on a monthly basis
            for year in np.arange(yr1, yr2+1):                                          # loop over years
                try:
                    annual_data = data['TAVG'].loc[str(year)+'-01-01':str(year)+'-12-01']   # get average temperature data (TAVG) for one year
                    df_annual[year] = annual_data.values
                except:                                                                 # in case of incomplete data for one year
                    break                                                               # break the loop
        return yr1, yr2, df_annual 
    
    # =============================================================================
    # temp_binary
    # =============================================================================
    """this function binarizes the temperature data based on the temperature 
    difference between two corresponding values in two consecutive years."""
    def temp_binary(df_annual, year, interval):
        warm_thresh = 30                                                                # average temperature for a warm month
        cold_thresh = 15                                                                 # average temperature for a cold month
        heat_factor = 1.5                                                               # warmer summers have more effect on energy consumption than colder winters
        temp_py = df_annual.loc[:, year-interval : year-1].mean(axis=1)                 # extract an average of previous values
        temp_cy = df_annual[year]                                                       # current year data
        temp_diff = temp_cy - temp_py                                                   # calculate temperature difference between current year and average of previous years
        cc_indicator = []                                                               # create a list to store 1 values corresponding to climate change risk
        if len(temp_diff) == 365:                                                       # if temp_diff is daily
            for i in range(len(temp_diff)):
                if i in np.arange(0,91):                                                # Jan-Feb-Mar (get 1 if it gets colder in winter)
                    temp_diff[i] = 1 if temp_diff[i] < 0 else 0
                if i in np.arange(91,274):                                              # Apr-May-Jun-Jul-Aug-Sep (get 1 if it gets warmer in summer)
                    temp_diff[i] = 1 if temp_diff[i] > 0 else 0
                if i in np.arange(274,365):                                             # Oct-Nov-Dec (get 1 if it gets colder in winter)
                    temp_diff[i] = 1 if temp_diff[i] < 0 else 0
        elif len(temp_diff) == 12:                                                      # if temp_diff is monthly
            for i in range(len(temp_diff)):
                """for Mar & Nov, if average temperature is higher than warm_
                thresh or lower than cold_thresh, then it is an indicator of 
                climate change and therefore, increased energy consumption:
                    add 1 to cc_indicator
                Note that there is no comparison of temperature in here.
                for Apr-Oct, if it gets warmer or the average temperature is 
                higher than warm_thresh, indicate climate change:
                    add 1 to cc_indicator.
                for Apr-Oct, if it gets colder or the average temperature is 
                lower than cold_thresh, indicate climate change:
                    add 1 to cc_indicator"""
                if (i in [3,11]):
                    if (temp_cy[i] >= warm_thresh) or (temp_cy[i] <= cold_thresh):      # Mar & Nov (neutral months)
                        cc_indicator.append(1)
                    else: cc_indicator.append(0)
                elif i in np.arange(4,11):                                              # Apr-May-Jun-Jul-Aug-Sep-Oct (get 1 if it gets warmer in summer)
                    if ((temp_diff[i] > 0) or temp_cy[i] >= warm_thresh):
                        cc_indicator.append(1*heat_factor)
                    else: cc_indicator.append(0)
                elif i in [12,1,2]:                                                     # Dec-Jan-Feb (get 1 if it gets colder in winter)
                    if ((temp_diff[i] < 0) or temp_cy[i] <= cold_thresh):
                        cc_indicator.append(1)
                    else: cc_indicator.append(0)
                    
        ### reformat cc_indicator as binary
        # pos = int(np.round(np.sum(cc_indicator)))
        # neg = cc_indicator.count(0)
        # cc_indicator = np.concatenate((np.ones(pos),np.zeros(neg)))
        
        return cc_indicator
    
    temp_data = pd.read_csv(temp_file)                                                  # read temperature data
    yr1, yr2, df_annual = annual_data(temp_data, 'm')                                   # aggregate data based on annual average temperature
    prior = stats.beta.pdf(x = initial_risk, a=a0, b=b0)                                # calculate the pdf for observed risk values
    df_risk = pd.DataFrame()                                                            # create a dataframe to store agents' risk values for each year
    for year in np.arange(yr1+1, yr2):                                                  # loop through years; for future simulation: 2021-2050
        df_risk[year] = initial_risk                                                    # save initial risk to dataframe
        cc_indicator = temp_binary(df_annual, year, 1)                                  # binarize temperature difference between year a moving window average
        a0 = a0+np.sum(cc_indicator)                                                    # update alpha parameter for posterior model
        b0 = b0+len(cc_indicator)-np.sum(cc_indicator)                                  # update beta parameter for posterior model
        secondary_risk = np.random.beta(a0, b0, size=n_agt)                             # secondary risk values for agents                                        
        posterior = stats.beta.pdf(x = secondary_risk, a=a0, b=b0)                      # calculate the pdf for secondary risk values
        updated_risk = (initial_risk*prior+secondary_risk*posterior)/(prior+posterior)  # update risk value based on weighted average
        initial_risk = updated_risk                                                     # update risk values
        prior = stats.beta.pdf(x = updated_risk, a=a0, b=b0)                            # update prior
    
    df_risk.to_csv(os.path.join(dst,'df_risk.csv'), index=False)
    return df_risk                   

# df_rsr_file = r'C:\Projects\ABM_PGMS\ABM\Future_Data\dist_RSR_Aug-09 13-55_2.csv'
# df_rsr = pd.read_csv(df_rsr_file)
# initial_risk = np.array(df_rsr["risk"])
# n_agt = 161
# a0 = 5
# b0 = 5
# temp_file = r'C:\Projects\ABM_PGMS\ABM\Future_Data\Climate_Scenarios\deltaT=6.csv'
# df_risk, cc_indicator = update_risk(temp_file, n_agt, initial_risk, a0, b0)

# =============================================================================
# read_data
# =============================================================================
"""this function reads data from a *.csv file"""
def read_data(file_path):                                                               
    df = pd.read_csv(file_path, header= 0, delimiter= ",")                              # read *.csv file.
    return df                                                                           # return dataframe
    
# =============================================================================
# read_one_year_data
# =============================================================================
"""this function reads "demand" or "price" data for one year"""
def read_one_year_data(LZ, text, year):
    if text == "demand":                                                                # select folder to read "demand" data
        folder = r'C:\Projects\ABM_PGMS\ABM\Future_Data\Demand'
    elif text == "price":                                                               # select folder to read "price" data
        folder = r'C:\Projects\ABM_PGMS\ABM\Future_Data\Price'
    file = os.path.join(folder, text + "_" + str(year) + ".csv")                        # create full file path to read data
    data = pd.read_csv(file, index_col= 0, thousands=',').T                             # read data from *.csv file and transpose to match the format
    data = data[LZ][0:12]                                                               # select only data for 6 load zones and 12 months
    data.insert(0, 'Year', year)                                                        # insert a column as "Year"
    
    return data                                                                         # return data    

# =============================================================================
# supply_curve
# =============================================================================
"""this function outputs a linear regression model as price = a*demand + b"""
def supply_curve(LZ, year):
    demand_data = read_one_year_data(LZ, "demand", year)                                # read demand data for one year
    price_data = read_one_year_data(LZ, "price", year)                                  # read price data for one year
    linear = []                                                                         # empty list to store [slope, intercept] for each year
    for lz in LZ:                                                                       # loop over load zones
        X = demand_data[lz].values                                                      # read demand data as X
        x_train = X.reshape(-1, 1)                                                      # reshape X data
        y = price_data[lz].values                                                       # read price data as y
        # correct anomaly 
        for j in range(len(y)):                                                         # loop over price values
            p = y[j]                                                                    # price in month j
            if p >= 50:                                                                 # if price is larger than $50/MWh
                y[j] = 50                                                               # set price upper bound to $50/MWh
            elif p <= 0:
                y[j] = 0                                                                # set negative price to 0
    
        reg = LinearRegression().fit(x_train,y)                                         # create a linear regression model
        linear.append([reg.coef_[0],reg.intercept_])                                    # [slope, intercept] of the linear supply curve
    
    return linear                                                                       # return a list of [slope, intercept] values for one year for 6 load zones

# =============================================================================
# PBP
# =============================================================================
"""this function returns a single value for PBP based on an array of cash_flow 
and lifespan"""
def PBP(G_cash_flow, G_ls):
    a = np.arange(G_ls+1)
    b = np.cumsum(G_cash_flow)
    if max(b)> 0:                                                                       # positive cash_flow means a good investment
        f = interpolate.interp1d(b,a)                                                   # interpolate f based on a linear function
        pbp = float(f(0))
    else:
        pbp = 999                                                                       # negative cash_flow means a bad investment
    return pbp                                                                          # return PBP (PayBack Period)

# =============================================================================
# NPV
# =============================================================================
"""this function returns single value for NPV based on an array of IRR, 
cash_flow, and lifespan"""
def NPV(G_IRR, G_cash_flow, G_ls):                                                      # Gtype: 0 NG, 1 Solar, 2 Wind
    npv = 0                                                                             # initial Net Present Value
    for i in np.arange(1,G_ls+1):                                                       # loop over the years in lifespan
        npv = npv + G_cash_flow[i]/((1+np.array(G_IRR))**int(i))                        # update NPV iteratively
    return npv                                                                          # return NPV (Net Present Value) 

# =============================================================================
# evaluate
# =============================================================================
"""this function calculates the profitability of a technology based on IRR"""
def evaluate(p, cf, cost, G_ls, G_tech): 
    G = generator(G_tech, cost, int(G_ls))                                              # assign G to generator class with attributes: technology, cost ($/MW), life span
    x_invest = float(cost)                                                              # initial investment = installation cost 
    G.cf = cf                                                                           # append capacity factor to generator
    G.capacity = 1                                                                      # append capacity (equal to 1 MW) to generator object
    ### with O&M costs
    if G_tech == 'NG':
        variable_cost = 2.56                                                            # $/MW EIA 2020 cost data; assume fixed
        # annual cash flow from generation (MW/h) multiplied by capacity factor
        # considering fuel and O&M costs as variables_cost
        # 8760 = 24*365 hours in one year
        # 0.01: capacity factor is in % format
        cash_flow = list(G.capacity*(p - variable_cost)*8760*G.cf*0.01*
                          np.ones(G.life_span))
        cash_flow.insert(0,-x_invest)                                                   # installation cost is considered as initial investment (negative)
    else:                                                                               # no fuel or O&M cost for renewable technologies
        cash_flow = list(G.capacity*p*8760*G.cf*0.01*np.ones(G.life_span))
        cash_flow.insert(0,-x_invest)                                                   # installation cost is considered as initial investment (negative)
    
    # ### without O&M costs
    # cash_flow = list(G.capacity*p*8760*G.cf*0.01*np.ones(G.life_span))                  # calculate cashflow
    # cash_flow.insert(0,-x_invest)                                                       # installation cost is considered as initial investment (negative)
    
    G.cash_flow = cash_flow                                                             # append cash flow data to generator object
    G.IRR = round(npf.irr(cash_flow),4)                                                 # calculate IRR and append it to generator object
    G.NPV = NPV(G.IRR, G.cash_flow, G.life_span)                                        # append NPV to generator object 
    # G.PBP = PBP(G.cash_flow, G.life_span)                                               # append data to generator object 
    
    return G                                                                            # return generator object with cash_flow, IRR, NPV

# =============================================================================
# ZoneInvest
# =============================================================================
"""this functions returns a technology type for investment (with the highest IRR) 
and the corresponding IRR"""
def ZoneInvest(new_P, rec_p, df_G):
    G_name = df_G.columns                                                               # names of the generation technologies
    G_IRR = []                                                                          # Internal Rate of Return
    G_NPV = []                                                                          # Net Present Value
    # G_PBP = []                                                                          # payback period
    
    for G_tech in G_name:
        if G_tech == 'NG':                                                              # no added price for non-renewable technology 
            p = new_P                                                                   # price = predicted price
        else:                                                                           # added price for renewable technologies
            p = (new_P + rec_p)                                                         # price = predicted price + carbon credit price
        cf = df_G[G_tech]['CF']                                                         # capacity factor
        cost = df_G[G_tech]['Cost']                                                     # installation cost (this model only considers installation costs)
        G_ls = df_G[G_tech]['LS']                                                       # life span
        """based on price, cost, and capacity factor, "evaluate" decides the
        best investment decision and returns IRR, NPV, and PBP for that decision"""
        G = evaluate(p, cf, cost, G_ls, G_tech)
        G_IRR.append(G.IRR)                                                             # append IRR to generator
        # G_PBP.append(G.PBP)                                                           # append PBP to generator
        G_NPV.append(G.NPV)                                                             # append NPV to generator
    
    max_IRR = max(G_IRR)                                                                # maximum IRR of the generation technologies   
    G_i = G_IRR.index(max_IRR)                                                          # index of the best investment technology
    G_tech = G_name[G_i]                                                                # best investment technology
    
    return G_tech, max_IRR                                                              # return best investment technology and the IRR

# =============================================================================
# AgentInvest
# =============================================================================
"""this function determines the technology and amount of investment in each 
load zone (LZ). Based on max_IRR determined from "ZoneInvest", if for multiple 
LZs: IRR>threshold, agents will invest in all of them proportional to the IRRs. """
def AgentInvest(LZ, IRR_t_array, capacity_deficit, IRR_threshold):
    lz_index = list(np.where(IRR_t_array > IRR_threshold)[0])                           # identify the investment technologies that have IRR>IRR_threshold 
    lz_irr_perc = np.zeros([1,len(LZ)])[0]                                              # initial values for investment proportions

    if len(lz_index) > 0:                                                               # if multiple LZs exceed the IRR_threshold
        lz_irr_perc[lz_index] = IRR_t_array[lz_index]/sum(IRR_t_array[lz_index])        # the agent invests in them proportional to the IRRs
    else:                                                                               # if no LZs exceed the IRR_threshold
        lz_index = list(np.where(IRR_t_array == IRR_t_array.max())[0])                  # identify LZ with max_IRR
        lz_irr_perc[lz_index] = 1                                                       # the agent only invests in the LZ with the max_IRR
    lz_invest = {}                                                                      # create a dict to store investment decision values
    for i in range(len(LZ)): 
        if i in lz_index:                                                               # if the selected LZ exceeds IRR_threshold  
            lz_invest[LZ[i]] = lz_irr_perc[i]*capacity_deficit                          # make an investment proportinal to the IRRs
        else:                                                                           # if the selected LZ does not exceed the IRR_threshold
            lz_invest[LZ[i]] = 0.0                                                      # make no investment
    
    return lz_invest                                                                    # return a dict for investment decisions in each LZ

# =============================================================================
# AgentInvestCF
# =============================================================================
"""this function determines the technology and amount of investment in each 
load zone (LZ). Based on max_IRR determined from "ZoneInvest", if for multiple 
LZs: IRR>threshold, agents will invest in all of them proportional to the IRRs.
This code also implements CF from the selected technology. The agent will 
invest less propotional to the CF because a more efficient technology can 
output more power so there is no need to overinvest."""
def AgentInvestCF(LZ, IRR_t_array, capacity_deficit, IRR_threshold, df_G, tech_row):
    lz_index = list(np.where(IRR_t_array > IRR_threshold)[0])                           # identify the investment technologies that have IRR>IRR_threshold 
    lz_irr_perc = np.zeros([1,len(LZ)])[0]                                              # initial values for investment proportions

    if len(lz_index) > 0:                                                               # if multiple LZs exceed the IRR_threshold
        lz_irr_perc[lz_index] = IRR_t_array[lz_index]/sum(IRR_t_array[lz_index])        # the agent invests in them proportional to the IRRs
    else:                                                                               # if no LZs exceed the IRR_threshold
        lz_index = list(np.where(IRR_t_array == IRR_t_array.max())[0])                  # identify LZ with max_IRR
        lz_irr_perc[lz_index] = 1                                                       # the agent only invests in the LZ with the max_IRR
    lz_invest = {}                                                                      # create a dict to store investment decision values
    for i in range(len(LZ)): 
        if i in lz_index:                                                               # if the selected LZ exceeds IRR_threshold
            CF_tech = df_G.loc['CF', tech_row[LZ[i]+'_tech']]/100                       # determine the CF for the best technology in the LZ
            lz_invest[LZ[i]] = lz_irr_perc[i]*capacity_deficit/CF_tech                  # make an investment proportinal to the IRR and CF
        else:                                                                           # if the selected LZ does not exceed the IRR_threshold
            lz_invest[LZ[i]] = 0.0                                                      # make no investment
    
    return lz_invest                                                                    # return a dict for investment decisions in each LZ

# =============================================================================
# generate_ABM_samples
# =============================================================================
"""this function generates samples for agents' perception of cost, REC, and risk 
(hesitation)"""
def generate_samples(dist_params, samples_dir, renewable, n_agt, date_now, k):          # dist_params = [mu_wind, sd_wind, mu_solar, sd_solar]
    new_dir = os.path.join(samples_dir, date_now)                                       # samples_dir = r'C:\Projects\ABM_PGMS\ABM\Calibration'
    isExist = os.path.exists(new_dir)                                                   # Check whether the specified path exists or not
    if not isExist:  
        os.makedirs(new_dir)                                                            # create new directory to store new samples
        # print("New directory is created!")                                              

    """generate random distribution for cost, risk, and company size;
    distribution parameters are determined by calibration; here, the results
    of the calibration have been hard-coded into the program.
    cost adjustment: companies operating on a specific technology could do
    the job at a lower cost than expected (familiarity with the technology)"""
    ng_cost_dist = list(np.ones([1,renewable])[0]) + \
        list(np.random.normal(dist_params[0],dist_params[1],n_agt-renewable))           # cost adjustment for non-renewable companies
    solar_cost_dist = list(np.random.normal(dist_params[2],dist_params[3],renewable)) \
        + list(np.ones([1,n_agt-renewable])[0])                                         # cost adjustment for solar companies
    solar_cost_dist = [abs(item) for item in solar_cost_dist]                           # make negative values positive (if any)
    wind_cost_dist = list(np.random.normal(dist_params[4],dist_params[5],renewable)) \
        + list(np.ones([1,n_agt-renewable])[0])                                         # cost adjustment for wind companies


    s = np.random.normal(1, 0.2, n_agt)                                                 # REC distribution for all agents
    """assumption: larger companies can get more carbon credit because they have
    larger market share and resources; so, data is sorted to attribute larger
    REC values to larger companies"""
    rec_dist = np.sort(s)
    # agents are risk-averse: they will invest less than required (hesitation)
    # agt_risk_f = np.random.normal(dist_params[4], dist_params[5], n_agt)                # risk Normal distribution for all agents
    agt_risk_f = np.random.beta(dist_params[6], dist_params[7], n_agt)                  # risk Beta distribution for all agents 
    sample_agt = np.random.gamma(shape=1.3, scale=3, size=n_agt)                        # size distribution for all agents
    agt_size_dist = np.sort(sample_agt/sum(sample_agt))                                 # agents' ranked captial distribution in the market       
    
    # save the samples
    dist_cost = {"NG": ng_cost_dist , "Solar": solar_cost_dist, 
                      "Wind": wind_cost_dist}                                           # a dict for cost distributions
    dist_RSR = {"risk": agt_risk_f, "size": agt_size_dist, "REC": rec_dist}             # a dict for agents' risk, size, and REC
    df1 = pd.DataFrame.from_dict(dist_cost)                                             # save the dict into a dataframe  
    file1 = os.path.join(new_dir, "dist_cost_" + date_now + "_" + str(k) + ".csv")      # create full file path for saving
    df1.to_csv (file1, index = False, header=True)                                      # save dataframe to *.csv file
    df2 = pd.DataFrame.from_dict(dist_RSR)                                              # save the dict into a dataframe
    file2 = os.path.join(new_dir, "dist_RSR_" + date_now + "_" + str(k) + ".csv")       # create full file path for saving
    df2.to_csv (file2, index = False, header=True)                                      # save dataframe to *.csv file
    
    return dist_cost, dist_RSR                                                          # return two dicts for 1- cost;  2- risk, size, REC

# =============================================================================
# aggregate
# =============================================================================
"""this function aggregates the results by technology, year, and load zone"""
def aggregate(n_agt, result, Retire, cap_initial):
    row_list = []                                                                       # create an empty list to store aggregation results  
    for agt_i in range(n_agt):
        subset = result[result["Agent"] == str(agt_i)]                                  # select only the results for agt_i
        subset_invest = subset.iloc[:,0:6]                                              # investment in Load Zones
        subset_tech = subset.iloc[:,6:]                                                 # year, agent ID, technology invested in Load Zones
        for y in np.arange(len(subset_tech.index)):                                     # loop through years
            # initial investment values    
            invest_tech = {'Year': subset.iloc[y,6], 'Agent': agt_i,                    # for Calibration: base_year = 2012
                           'NG': 0.0,'Solar': 0.0,'Wind': 0.0}                          # for Future Simulation: base_year = 2021           
            for lz in range(6):                                                         # loop through 6 loadzones
                if subset_tech.iloc[y, lz+2] == "NG":
                    invest_tech["NG"] += subset_invest.iloc[y, lz]                      # update the investment value for NG 
                elif subset_tech.iloc[y, lz+2] == "Solar":      
                    invest_tech["Solar"] += subset_invest.iloc[y, lz]                   # update the investment value for Solar  
                elif subset_tech.iloc[y, lz+2] == "Wind":     
                    invest_tech["Wind"] += subset_invest.iloc[y, lz]                    # update the investment value for Wind
                else: 
                    print("Something went wrong!")
            row_list.append(invest_tech)                                                # add the updated data to row_list                   

    agg_tech = pd.DataFrame(row_list)                                                   # create a dataframe for investment aggregated by technology types
    agg_year = agg_tech.groupby("Year").sum()                                           # sum the investment by technology types for each year
    del agg_year["Agent"]                                                               # delete the agt ID column
    agg_year_retire = agg_year.copy()                                                   # make a copy of agg_year to store results without retirement
    for i in range(len(agg_year_retire['NG'])):                                         # update the results with retirement
        agg_year_retire.iloc[i,0] = max(agg_year_retire.iloc[i,0]- Retire.values[i], 0) # subtract retirement from NG (max is to avoid negative values)
    
    agg_year["Total"] = agg_year.sum(axis=1)                                            # add a column for total investment
    agg_year["Total_cum"] = agg_year["Total"].cumsum() + cap_initial                    # calculate cumulative capacity
    agg_year_retire["Total"] = agg_year_retire.sum(axis=1)                              # add a column for total investment
    agg_year_retire["Total_cum"] = agg_year_retire["Total"].cumsum() + cap_initial      # calculate cumulative capacity
    subset = result.iloc[:,0:7]                                                         # investment in Load Zones
    agg_LZ = subset.groupby("Year").sum()                                               # sum the investment by technology types for each year
    return agg_tech, agg_year, agg_year_retire, agg_LZ                                  # return three dataframes: aggregation by technology, year, and load zone

# =============================================================================
# KGE_stat
# =============================================================================
"""this function calculates the KGE value of the simulated and observed data."""
def KGE_stat(tech_yr, tot_new_ca, cap_hist):
    wind_cumsum_hist = np.cumsum(cap_hist['Wind'])                                      # calculate the cumulative sum of Wind installation
    solar_cumsum_hist = np.cumsum(cap_hist['Solar'])                                    # calculate the cumulative sum of Solar installation
    wind_cumsum = np.cumsum(tech_yr['Wind'])                                            # calculate the cumulative sum of simulated Wind installation
    solar_cumsum = np.cumsum(tech_yr['Solar'])                                          # calculate the cumulative sum of simulated Solar installation
    ## Pearson Correlation Coefficient
    r_w, p_w = scipy.stats.pearsonr(wind_cumsum, wind_cumsum_hist)
    r_s, p_s = scipy.stats.pearsonr(solar_cumsum, solar_cumsum_hist)
    r_tot, p_tot = scipy.stats.pearsonr(tot_new_ca, cap_hist['Total'])
    ## Mean Ratios
    alpha_w = np.mean(wind_cumsum)/np.mean(wind_cumsum_hist)                            # wind mean ratio
    alpha_s = np.mean(solar_cumsum)/np.mean(solar_cumsum_hist)                          # solar mean ratio
    alpha_tot = np.mean(tot_new_ca)/np.mean(cap_hist['Total'])                          # total generation capacity mean ratio
    ## Standard Deviation Ratios
    beta_w = np.std(wind_cumsum)/np.std(wind_cumsum_hist)                               # wind standard deviation ratio
    beta_s = np.std(solar_cumsum)/np.std(solar_cumsum_hist)                             # solar standard deviation ratio
    beta_tot = np.std(tot_new_ca)/np.std(cap_hist['Total'])                             # total generation standard deviation ratio
    ## Kling-Gupta Efficiency scores (KGE)
    KGE_w = 1 - np.sqrt((1-r_w)**2+(1-alpha_w)**2+(1-beta_w)**2)                        # KGE for simulated Wind
    KGE_s = 1 - np.sqrt((1-r_s)**2+(1-alpha_s)**2+(1-beta_s)**2)                        # KGE for simulated Solar
    KGE_tot = 1 - np.sqrt((1-r_tot)**2+(1-alpha_tot)**2+(1-beta_tot)**2)                # KGE for total simulated capacity

    KGE = {'KGE_Wind':KGE_w, 'KGE_Solar':KGE_s, 'KGE_Total':KGE_tot}                    # Peason's Coefficient of Correlation
    p = {'p_value_Wind':p_w, 'p_value_Solar':p_s, 'p_value_Total':p_tot}                # p-value for hypothesis testing

    return KGE, p

