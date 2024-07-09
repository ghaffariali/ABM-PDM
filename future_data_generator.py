# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 13:04:13 2022

@author: Ali Ghaffari       alg721@lehigh.edu

this code creates future data for 'Cost', 'CF', and 'Retirement' based on 
annual change rate.
"""
# import libraries
import os
import pandas as pd
import numpy as np

# =============================================================================
# future_data_gen
# =============================================================================
def future_data_gen(rate, start, end, output_dir, key):    
    df = pd.DataFrame()                                                                 # create a df to store results
    df["Year"] = np.arange(start, end+1)                                                # fill in Year values
    if key in ['Cost','CF']:
        if key=='Cost':
            base_vals = {'ng':1661000, 'solar':1420000, 'wind':1462330}                 # base values for Cost from 2020
        elif key=='CF':
            base_vals = {'ng':56.6, 'solar':24.9, 'wind':35.4}                          # base values for CF from 2020
        ng = [base_vals['ng']]                                                          # create lists to store results
        solar = [base_vals['solar']]
        wind = [base_vals['wind']]
        for y in range(len(df)):                                                        # loop over years
            # ng.append(ng[y]*(1.-rate))
            solar.append(solar[y]*(1.+rate))                                            # calculate for solar
            wind.append(wind[y]*(1.+rate))                                              # calculate for wind
        df["NG"] = base_vals['ng']                                                      # NG is assummed to be constant
        df["Solar"] = solar[1:]
        df["Wind"] = wind[1:]
    else:
        base_retire = 1713.1                                                            # base retirement from 2020
        df['Retirement'] = base_retire                                                  # retirement is assumed to be constant
    if rate >= 0:                                                                       # create output file name
        output_name = '{}_{:.2f}_pect_increase.pkl'.format(key,abs(rate*100))
    elif rate < 0:
        output_name = '{}_{:.2f}_pect_decrease.pkl'.format(key,abs(rate*100))
        
    df.set_index('Year', inplace=True)                                                  # set Year as index
    df.to_pickle(os.path.join(output_dir, key, output_name))                            # save df as *.pkl
    df.to_csv(os.path.join(output_dir, key, output_name[:-3]+'csv'))                    # save df as *.csv
    
    return df                                                                           # return df

# rate_list = [0.0, -0.015, -0.03, 0.0, 0.015, 0.03, 0.0]                                 # desired rates
rate_list = [-0.01, -0.02, 0.01, 0.02, 0.0]                                             # desired rates to address Reviewer 3's comment
# key_list = ['Cost', 'Cost', 'Cost', 'CF', 'CF', 'CF', 'Retirement']                     # variable keys
key_list = ['Cost', 'Cost', 'CF', 'CF', 'Retirement']                                   # variable keys to address Reviewer 3's comment
output_dir = r'C:\Projects\ABM_PGMS\ABM\Future_Data'                                # output directory

# remove old files
for key in ['Cost', 'CF', 'Retirement']:
    old_files = [f.path for f in os.scandir(os.path.join(output_dir,key)) 
                 if f.is_file()]                                                        # list old files
    for f in old_files:
        os.remove(f)

for i in range(len(rate_list)):
    future_data_gen(rate_list[i], 2021, 2050, output_dir, key_list[i])
    
    