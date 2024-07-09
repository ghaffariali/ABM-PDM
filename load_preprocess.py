# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 20:33:11 2022

@author: Ali Ghaffari           alg721@lehigh.edu

this code preprocesses load input data for GCAM and ERCOT. The input is raw
load for GCAM and ERCOT in tabular format; the output is annual load for GCAM
and ERCOT in vectorized format with hour 0 added.
"""
# import libraries
import os
import numpy as np
import pandas as pd

#%% function
# =============================================================================
# raw2base
# =============================================================================
"""this function converts raw load into base load based on GCAM loads:
    1- GCAM (default): interpolates GCAM load profiles between two decades to 
    get annual GCAM load profiles. Interpolation is simple and done by 
    np.linspace. the code returns annual GCAM load for each scenario as a 
    vector with hour 0 included.
    2- ERCOT: redistributes ERCOT annual load over GCAM hourly load as a vector
    with 0 hour included.""" 
def raw2base_gcam(loads_raw_dir, n, loads_base_dir, d=10, ercot_load_file="", source="gcam"):
    if os.path.exists(loads_base_dir) == False:                                         # check if output_dir exists; if not, create one.
        os.mkdir(loads_base_dir)
    if ercot_load_file != "": 
        ercot_load = pd.read_csv(ercot_load_file, usecols=["Year", "Energy (MWh)"], 
                                 index_col="Year")                                      # read ERCOT future load from *.csv
    zero_load = pd.DataFrame(np.zeros([1,123]))                                         # create a row of zeros as initial load
    load_scs = [f.path for f in os.scandir(loads_raw_dir) if f.is_dir()]                # get directories of load scenarios
    for load_sc in load_scs:                                                            # loop over load scenarios
        load_yrs = [f.path for f in os.scandir(load_sc) if f.is_dir()]                  # get directories of load years (2030s, 2040s, 2050s, etc.)
        load_prs = os.listdir(load_yrs[0])                                              # list load profiles
        output_sc = os.path.join(loads_base_dir, os.path.basename(load_sc))             # output folder with the same name as load_sc
        try:                                                                            # if the folder does not already exist
            os.mkdir(output_sc)                                                         # create a folder for load_sc
        except OSError as error:                                                        # if folder exists
            print(error)                                                                # print error message
        for i in range(n):                                                              # loop over number of decades
            for load_pr in load_prs:                                                    # loop over load profiles
                yr1 = pd.read_csv(os.path.join(load_yrs[i], load_pr), 
                                  delimiter="\t", header=None)                          # read load profile from folder1 as a list
                yr2 = pd.read_csv(os.path.join(load_yrs[i+1], load_pr), 
                                  delimiter="\t", header=None)                          # read load profile from folder2 as a list
                annual_loads = np.linspace(yr1, yr2, d+1)                               # interpolate 10 values to get annual loads between 10 years
                for k in range(np.size(annual_loads, 0)):                               # loop over years
                    year = 2021 + d*i + k                                               # year
                    try: 
                        os.mkdir(os.path.join(output_sc, str(year)))                    # create a folder with the same name as year
                    except OSError:
                        print('directory exists!')
                    new_load = pd.DataFrame(annual_loads[k,:,:])                        # create a dataframe from annual load                    
                    base_load = pd.concat([zero_load, new_load], ignore_index=True)     # concatenate two dataframes
                    base_load = base_load.transpose().values.reshape(-1,1)              # reshape the dataframe to add as a column to frame
                    if source=="ercot":                                                 # if load source is ERCOT
                        base_load = base_load*ercot_load.loc[year]['Energy (MWh)']\
                            /np.sum(base_load)/365                                      # multiply load by annual ERCOT load and divide by 365
                    base_load = pd.DataFrame(base_load)                                 # create a dataframe out of a list
                    base_load.to_csv(os.path.join(output_sc, str(year), load_pr),
                              sep="\t", index=False, header=False)                      # save annual load data as *.txt file

# =============================================================================
# raw2base_ercot
# =============================================================================
"""this function converts raw load into base load based on ERCOT loads by 
interpolating ERCOT load profiles between two periods to get annual ERCOT load 
profiles. Interpolation is simple and done by np.linspace. the code returns 
annual ERCOT load for each scenario as a vector with hour 0 included."""
def raw2base_ercot(loads_raw_dir, loads_base_dir, d=5, zeroLoad=False):
    if os.path.exists(loads_base_dir) == False:                                         # check if output_dir exists; if not, create one.
        os.mkdir(loads_base_dir)
    zero_load = pd.DataFrame(np.zeros([1,123]))                                         # create a row of zeros as initial load
    load_yrs = [f.path for f in os.scandir(loads_raw_dir) if f.is_dir()]                # get directories of load years (2021-2025, 2026-2030, etc.)
    
    for i in range(len(load_yrs)-2):                                                      # loop over periods
        dir1 = os.listdir(load_yrs[i])                                                  # dir of folder1
        dir2 = os.listdir(load_yrs[i+1])                                                # dir of folder2
        for p in range(len(dir1)):                                                      # loop over folder files (load profiles)
            yr1 = pd.read_csv(os.path.join(load_yrs[i], dir1[p]), 
                              sep=" ", header=None)                                     # read load profile from folder1 as a list
            yr2 = pd.read_csv(os.path.join(load_yrs[i+1], dir2[p]), 
                              sep=" ", header=None)                                     # read load profile from folder2 as a list
            annual_loads = np.linspace(yr1, yr2, d+1)                                   # interpolate d values to get annual loads between years
            for k in range(np.size(annual_loads, 0)):                                   # loop over years
                year = 2021 + d*i + k                                                   # year
                try: 
                    os.mkdir(os.path.join(loads_base_dir, str(year)))                   # create a folder with the same name as year
                except OSError: 
                    print('directory exists!')
                new_load = pd.DataFrame(annual_loads[k,:,:])                            # create a dataframe from annual load                    
                if zeroLoad:                                                            # add initial load
                    base_load = pd.concat([zero_load, new_load.T], ignore_index=True)   # concatenate two dataframes
                else:                                                                   # do not add initial load
                    base_load = new_load.T
                base_load = base_load.transpose().values.reshape(-1,1)                  # reshape the dataframe to add as a column to frame
                base_load = pd.DataFrame(base_load)                                     # create a dataframe out of a list
                base_load.to_csv(os.path.join(loads_base_dir, str(year), dir1[p]),
                          sep="\t", index=False, header=False)                          # save annual load data as *.txt file
 
# =============================================================================
# rename_loads                                                         
# =============================================================================
"""this little function corrects names of load profiles. 
wrong name: P1load_wd_pk_q1.txt
right name: load_wd_pk_q1.txt
Check before running the code. If the names are in correct format, do not run 
this function."""
def rename_loads(loads_raw_dir):
    load_yrs = [f.path for f in os.scandir(loads_raw_dir) if f.is_dir()]
    for load_yr in load_yrs:
        loads = os.listdir(load_yr)
        for load in loads:
            filename_old = os.path.join(load_yr, load)
            filename_new = os.path.join(load_yr, load[2:])
            os.rename(filename_old, filename_new)
    print('Renaming done!')        


# directories and files
loads_raw_dir = r'C:\Projects\ABM_PGMS\ABM\Load_Data\Loads_1Raw\ERCOT'
loads_base_dir = r'C:\Projects\ABM_PGMS\ABM\Load_Data\Loads_2Base_ERCOT (Direct Method)'

# rename_loads(loads_raw_dir)
raw2base_ercot(loads_raw_dir, loads_base_dir)

