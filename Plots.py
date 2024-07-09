# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 09:55:54 2022

@author: Ali Ghaffari           alg721@lehigh.edu

this code outputs plots for the ABM-PGMS project.
"""
# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# functions
# =============================================================================
# aggregate
# =============================================================================
def aggregate(result, n_agt=161, cap_initial=128947, agg_type='year', tech=None):
    row_list = []                                                                       # create an empty list to store aggregation results  
    for agt_i in range(n_agt):
        subset = result[result["Agent"] == agt_i]                                       # select only the results for agt_i
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
    if agg_type=='year':
        agg_year = agg_tech.groupby("Year").sum()                                       # sum the investment by technology types for each year
        del agg_year["Agent"]                                                           # delete the agt ID column
        agg_year["Total"] = agg_year.sum(axis=1)                                        # add a column for total investment
        agg_year["Total_cum"] = agg_year["Total"].cumsum() + cap_initial                # calculate total cumulative capacity
        agg_year.reset_index(inplace=True)
    elif agg_type=='agent':
        df_mod = pd.DataFrame()
        for year in np.arange(2021,2051):                                               # reshape agent investments into annual data
            if tech!=None:
                df_mod[year] = agg_tech[tech][agg_tech['Year']==year].values            # sum investments for only one 'tech'
            else:
                df_mod[year] = agg_tech['NG'][agg_tech['Year']==year].values +\
                    agg_tech['Solar'][agg_tech['Year']==year].values +\
                        agg_tech['Wind'][agg_tech['Year']==year].values                 # sum all investments from all technologies
        agg_year = df_mod
    
    return agg_year                                                                     # return three dataframes: aggregation by technology, year, and load zone

# =============================================================================
# extract_lz
# =============================================================================
"""this function extracts investment data for each SA/CC and load scenario by 
year by loadzone and by technology. input: Total_Agent_Investment.csv"""
def extract_lz(scen_dir):
    frame = []                                                                          # list to append results
    # df_scen = pd.DataFrame()                                                            # df to store results
    LZ = ['LZ_AEN','LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST']               # load zones
    scens = [f.path for f in os.scandir(scen_dir) if f.is_dir()]                        # get directories of SA/CC scenarios
    for scen in scens:                                                                  # loop over SA/CC scenarios
        # print(os.path.basename(scen))
        load_scens = [f.path for f in os.scandir(scen) if f.is_dir()]                   # get directories of load scenarios
        for load_scen in load_scens:                                                    # loop over load scenarios
            df_scen = pd.DataFrame()                                                    # df to store results
            df = pd.read_csv(os.path.join(load_scen, 'Total_Agent_Investment.csv'))     # read the main result
            for lz in LZ:                                                               # loop over load zones
                df_lz = df[[lz, 'Year', lz+'_tech']]                                    # extract investment (MW), year, and technology for each loadzone
                for y in range(df_lz['Year'].min(), df_lz['Year'].max()+1):             # loop over years
                    df_lz_annual = df_lz[df_lz['Year'] == y]                            # filter df_temp by year
                    df_lz_tech = df_lz_annual.groupby(lz+'_tech').sum()                 # group by technology
                    for tech in ['NG', 'Solar', 'Wind']:                                # loop over technologies
                        if tech in df_lz_tech.index:                                    # if there is an investment for the technology
                            df_scen.loc[y,lz+'_'+tech] = df_lz_tech.loc[tech,lz]
                        else:                                                           # if no investment for the technology
                            df_scen.loc[y,lz+'_'+tech] = 0
            df_scen['Load_Scen'] = os.path.basename(load_scen)                          # save load scenario
            df_scen['Scen'] = os.path.basename(scen)                                    # save SA/CC scenario
            frame.append(df_scen)                                                       # append df to frame
    df_out = pd.concat(frame).reset_index().rename(columns={'index': 'Year'})
    
    return df_out, scens, load_scens
            
# =============================================================================
# extract_tech
# =============================================================================
"""this function reads 'Technology by Year' files from each scenario and load
scenario and outputs a df containing all data."""
def extract_tech(scen_dir):
    frame = []                                                                          # list to append results
    scens = [f.path for f in os.scandir(scen_dir) if f.is_dir()]                        # get directories of SA/CC scenarios
    for scen in scens:                                                                  # loop over SA/CC scenarios
        load_scens = [f.path for f in os.scandir(scen) if f.is_dir()]                   # get directories of load scenarios
        for load_scen in load_scens:                                                    # loop over load scenarios
            df = pd.read_csv(os.path.join(load_scen, 'Technology by Year.csv'))         # read investment for each technology by year
            df['Load_Scen'] = os.path.basename(load_scen)                               # save load scenario
            df['Scen'] = os.path.basename(scen)                                         # save SA/CC scenario
            frame.append(df)                                                            # append df to frame
    df_out = pd.concat(frame, ignore_index=True)                                        # concatenate all dfs
    
    return df_out, scens, load_scens

# =============================================================================
# extract_tech
# =============================================================================
"""this function reads 'Technology by Year' files from each scenario and load
scenario and outputs a df containing all data."""
def extract_total(scen_dir):
    frame = []                                                                          # list to append results
    scens = [f.path for f in os.scandir(scen_dir) if f.is_dir()]                        # get directories of SA/CC scenarios
    for scen in scens:                                                                  # loop over SA/CC scenarios
        load_scens = [f.path for f in os.scandir(scen) if f.is_dir()]                   # get directories of load scenarios
        for load_scen in load_scens:                                                    # loop over load scenarios
            df = pd.read_csv(os.path.join(load_scen, 'Total_Annual_Investment.csv'))    # read total investments and capacity deficit for each year
            df['Load_Scen'] = os.path.basename(load_scen)                               # save load scenario
            df['Scen'] = os.path.basename(scen)                                         # save SA/CC scenario
            frame.append(df)                                                            # append df to frame
    df_out = pd.concat(frame, ignore_index=True)                                        # concatenate all dfs
    
    return df_out, scens, load_scens

# =============================================================================
# extract_price
# =============================================================================
"""this function extracts price values from all PGMS outputs and calculates the
average for plotting."""
def extract_price(scen_dir):
    frame = []
    scens = [f.path for f in os.scandir(scen_dir) if f.is_dir()]                        # get directories of SA/CC scenarios
    for scen in scens:                                                                  # loop over SA/CC scenarios
        load_scens = [f.path for f in os.scandir(scen) if f.is_dir()]                   # get directories of load scenarios
        for load_scen in load_scens:                                                    # loop over load scenarios
            yrs = [f.path for f in os.scandir(os.path.join(load_scen, 'PGMS_Outputs')) 
                   if f.is_dir()]                                                       # get directories of annual results
            for yr in yrs:
                df = pd.DataFrame()
                load_profs = [f.path for f in os.scandir(
                    os.path.join(yr, '02_Prices')) if f.is_file()]                      # get all *.csv files for all load profiles
                for load_prof in load_profs:
                    df[os.path.basename(load_prof)] = pd.read_csv(load_prof)['Price']   # read price values for each load profile
                df['PAVG'] = df.mean(axis=1)                                            # take average of all prices for all load profiles
                frame.append([os.path.basename(scen), os.path.basename(load_scen),
                            os.path.basename(yr), df['PAVG'].mean()])                   # save PAVG to column yr in df_load_scen
    df_out = pd.DataFrame(data=frame, columns=['Scen', 'Load_Scen', 'Year', 'PAVG'])
    
    return df_out, scens, load_scens

# =============================================================================
# extract_price_lz
# =============================================================================
"""this function extracts price values from all PGMS outputs and calculates the
average for plotting."""
def extract_price_lz(scen_dir,bus_map_file):
    bus_map = pd.read_csv(bus_map_file)
    frame = []
    scens = [f.path for f in os.scandir(scen_dir) if f.is_dir()]                        # get directories of SA/CC scenarios
    for scen in scens:                                                                  # loop over SA/CC scenarios
        load_scens = [f.path for f in os.scandir(scen) if f.is_dir()]                   # get directories of load scenarios
        for load_scen in load_scens:                                                    # loop over load scenarios
            yrs = [f.path for f in os.scandir(os.path.join(load_scen, 'PGMS_Outputs')) 
                   if f.is_dir()]                                                       # get directories of annual results
            for yr in yrs:
                df = pd.DataFrame()
                load_profs = [f.path for f in os.scandir(
                    os.path.join(yr, '02_Prices')) if f.is_file()]                      # get all *.csv files for all load profiles                
                for load_prof in load_profs:
                    df[os.path.basename(load_prof)] = pd.read_csv(load_prof)['Price']   # read price values for each load profile
                df['PAVG'] = df.mean(axis=1)                                        # take average of all prices for all load profiles
                # df_out = df['PAVG']
                # df_out['LZ'] = bus_map['LZ_Name'].str.upper()                       # add LZ-Name as a columns to bus daily loads

                frame.append([os.path.basename(scen), os.path.basename(load_scen),
                            os.path.basename(yr), df['PAVG'].mean()])                   # save PAVG to column yr in df_load_scen
    df_out = pd.DataFrame(data=frame, columns=['Scen', 'Load_Scen', 'Year', 'PAVG'])
    
    return df_out, scens, load_scens

def extract_price_ls(scen_dir, base_loads_dir):
    frame = []
    scens = [f.path for f in os.scandir(scen_dir) if f.is_dir()]                        # get directories of SA/CC scenarios
    for scen in scens:                                                                  # loop over SA/CC scenarios
        load_scens = [f.path for f in os.scandir(scen) if f.is_dir()]                   # get directories of load scenarios
        for load_scen in load_scens:                                                    # loop over load scenarios
            yrs = [f.path for f in os.scandir(os.path.join(load_scen, 'PGMS_Outputs')) 
                   if f.is_dir()]                                                       # get directories of annual results
            for yr in yrs:
                df = pd.DataFrame()
                frame_load = []
                bl_files = [f.path for f in os.scandir(
                    os.path.join(base_loads_dir, os.path.basename(yr))) if f.is_file()] # get all base loads
                price_files = [f.path for f in os.scandir(
                    os.path.join(yr, '02_Prices')) if f.is_file()]                      # get all price files
                ls_files = [f.path for f in os.scandir(
                    os.path.join(yr, '03_load_shedding')) if f.is_file()]               # get all load shedding files
                
                for f in range(len(bl_files)):
                    load_name = os.path.basename(bl_files[f])[5:-4]                     # get name of load profile
                    ### find the correct file in ls_files and price_files
                    ls_file = [file for file in ls_files if load_name in os.path.basename(file)]
                    price_file = [file for file in price_files if load_name in os.path.basename(file)]
                    
                    if len(ls_file)>0:                                                  # if the solution for load profile exists
                        ##### calculate % of load satisfied
                        df_bl = pd.read_csv(bl_files[f], header=None)                   # read base load
                        df_ls = pd.read_csv(ls_file[0], header=None, sep=' ')           # read load shedding
                        load_p = (df_bl.values.sum()-
                                  df_ls.iloc[:,:-1].values.sum())/df_bl.values.sum()    # get the % of load that is satisfied
                        frame_load.append(round(load_p, 4))                             # append load_p to frame_load
                        
                        ##### calculate average price
                        df_p = pd.read_csv(price_file[0], usecols=['Price'])            # read Price column
                        df_p[(df_p['Price'] >= 50) | (df_p['Price'] < 0)] = np.nan      # replace >=50 and negative values with NaN
        
                        df[os.path.basename(price_file[0])] = df_p['Price']
                
                df['PAVG'] = df.mean(axis=1)                                            # take average of all prices for all load profiles
                frame.append([os.path.basename(scen), os.path.basename(load_scen),
                            os.path.basename(yr), df['PAVG'].mean(),
                            np.mean(frame_load)])                                       # save PAVG to column yr in df_load_scen
    
    df_out = pd.DataFrame(data=frame, columns=['Scen', 'Load_Scen', 'Year', 
                                               'PAVG', 'Load%'])                        # save frame as a dataframe
    
    return df_out, scens, load_scens

# =============================================================================
# count_no_sols
# =============================================================================
"""this function counts solutions for each SA combination"""
def count_no_sols(scen_dir):
    frame = []                                                                          # list to append results
    scens = [f.path for f in os.scandir(scen_dir) if f.is_dir()]                        # get directories of SA/CC scenarios
    for scen in scens:                                                                  # loop over SA/CC scenarios
        load_scens = [f.path for f in os.scandir(scen) if f.is_dir()]                   # get directories of load scenarios
        for load_scen in load_scens:                                                    # loop over load scenarios
            sol_dir = os.path.join(load_scen, 'PGMS_Outputs')
            count = len([f for f in os.scandir(sol_dir) if f.is_dir()])                 # count all directories (years with at least one solution for PGMS)   
            no_sol = [os.path.basename(scen), os.path.basename(load_scen), count]       # get a list of SA scenario, load scenario, count
            frame.append(no_sol)                                                        # append no_sol to list
    
    return frame

# =============================================================================
# combine
# =============================================================================
"""this function combines results from GCAM and ERCOT."""
def combine(dir1, dir2):
    df_out1, scens, load_scens = extract_tech(scen_dir1)                                # generate data for plots
    df_out1['Scen'] = df_out1['Scen'].apply(lambda x: x[5:])                            # remove the word GCAM from Scen
    df_out2, scens2, load_scens2 = extract_tech(scen_dir2)                              # generate data for plots
    df_out2['Scen'] = df_out2['Scen'].apply(lambda x: x[6:])                            # remove the word ERCOT from Scen
    df_out = pd.concat([df_out1, df_out2])                                              # concat two dfs
    load_scens.append('C:\Baseline')                                                    # append Baseline load scenario to load_scens
    
    return df_out, scens, load_scens

# =============================================================================
# extract_tech_total
# =============================================================================
"""this function reads 'Technology by Year' files from each scenario and load
scenario, sums up the investments from 2021 to 2050  and outputs a df 
containing all data."""
def extract_tech_total(scen_dir):
    frame = []                                                                          # list to append results
    scens = [f.path for f in os.scandir(scen_dir) if f.is_dir()]                        # get directories of SA/CC scenarios
    for scen in scens:                                                                  # loop over SA/CC scenarios
        load_scens = [f.path for f in os.scandir(scen) if f.is_dir()]                   # get directories of load scenarios
        for load_scen in load_scens:                                                    # loop over load scenarios
            df = pd.read_csv(os.path.join(load_scen, 'Technology by Year.csv'))         # read investment for each technology by year
            df_total = df.sum(axis=0)                                                   # sum investments
            frame.append([os.path.basename(scen), os.path.basename(load_scen),
                          df_total['NG'], df_total['Solar'], df_total['Wind']])         # append values to frame
    df_out = pd.DataFrame(frame)                                                        # concatenate all dfs
    
    return df_out


# directories and files
output_dir = r'C:\Projects\ABM_PGMS\Results\10_Plots'
sa_gcam_dir = r'C:\Projects\ABM_PGMS\Results\02_SA\GCAM\Jun-17 11-30 - Additional runs'
# sa_ercot_dir = r'C:\Projects\ABM_PGMS\Results\02_SA\ERCOT\Nov-11 10-55'
sa_ercot_dir = r'C:\Projects\ABM_PGMS\Results\02_SA\ERCOT\Jun-20 13-45 - Additional runs'
cc_ercot_dir_00 = r'C:\Projects\ABM_PGMS\Results\03_CC\ERCOT\0.0cost_0.0cf\Nov-28 17-20'
cc_ercot_dir_33 = r'C:\Projects\ABM_PGMS\Results\03_CC\ERCOT\3.0cost_3.0cf\Apr-17 11-37 - PDM'
cc_gcam_dir_00 = r''
cc_gcam_dir_33 = r'C:\Projects\ABM_PGMS\Results\03_CC\GCAM\3.0cost_3.0cf\Apr-26 18-38 - PDM'
bus_map_file = r'C:\Projects\ABM_PGMS\ABM\Future_Data\bus_map.csv'
base_loads_dir = r'C:\Projects\ABM_PGMS\ABM\Load_Data\Loads_2Base_ERCOT (Direct Method)'

#%% 01_SA_LoadScen: load scenario for SA
"""investments for ERCOT/GCAM in 9 SA scenarios and n load scenarios. 
there is one plot for each load scenario."""
scen_dirs = [sa_gcam_dir, sa_ercot_dir]
load_srcs = ['GCAM', 'ERCOT']
gcam_loads = ['Low Emission, High Population & GDP',
              'Low Emission, Low Population & GDP',
              'Reference Emission, High Population & GDP',
              'Reference Emission, Low Population & GDP']
cap_initial = {'NG':99849, 'Solar':3975, 'Wind':25123}                                  # initial capacities
result_dir = os.path.join(output_dir, '01_SA_LoadScen')                                 # create result folder if not already there
if os.path.exists(result_dir) == False:
    os.mkdir(result_dir)

plt_type = 'delta%'
for i in range(1):
    i=1
    load_src = load_srcs[i]                                                             # get load source
    df_out, scens, load_scens = extract_tech(scen_dirs[i])                              # generate data for plots
    for l in range(len(load_scens)):                                                    # loop over load scenarios
        df_load = df_out[df_out['Load_Scen'] == os.path.basename(load_scens[l])]        # filter data for load scenario
        fig, axs = plt.subplots(5,5)                                                    # plot 9 subplots for 9 SA scenarios
        if plt_type=='cum': 
            plt.setp(axs, ylim=(0,300))                                                 # set ylimit for all subplots
        elif plt_type=='delta':
            plt.setp(axs, ylim=(0,8.1))                                                   # set ylimit for all subplots

        s = 0
        for i in range(5):                                                              # loop over cost values
            for j in range(5):                                                          # loop over cf values
                df_scen = df_load[df_load['Scen'] == os.path.basename(scens[s])]        # filter data by SA scenario & convert to GW
                if plt_type == 'cum':
                    axs[i,j].bar(df_scen['Year'], df_scen['NG'].cumsum()/1000 + cap_initial['NG']/1000-1.713, 
                                 color='r', label="NG", hatch="------")                 # 1.713GW is the retired capacity each year
                    axs[i,j].bar(df_scen['Year'], df_scen['Solar'].cumsum()/1000+ cap_initial['Solar']/1000,
                                 color='y', label="Solar", hatch="......",
                                 bottom=df_scen['NG'].cumsum()/1000 + cap_initial['NG']/1000-1.713)
                    axs[i,j].bar(df_scen['Year'], df_scen['Wind'].cumsum()/1000 + cap_initial['Wind']/1000,
                                 color='b', label="Wind", hatch="//////", 
                                 bottom=df_scen['Solar'].cumsum()/1000 + cap_initial['Solar']/1000 + df_scen['NG'].cumsum()/1000 + cap_initial['NG']/1000-1.713)
                elif plt_type == 'delta':
                    axs[i,j].bar(df_scen['Year'], df_scen['NG']/1000, color='r', label='NG', hatch='------')
                    axs[i,j].bar(df_scen['Year'], df_scen['Solar']/1000, color='y', label='Solar', hatch='......', bottom=df_scen['NG']/1000)
                    axs[i,j].bar(df_scen['Year'], df_scen['Wind']/1000, color='b', label='Wind', hatch='//////', bottom=df_scen['NG']/1000+df_scen['Solar']/1000)
                elif plt_type=='delta%':
                    df_scen_mod = df_scen[['NG', 'Solar', 'Wind']].div(df_scen[['NG', 'Solar', 'Wind']].sum(axis=1), axis=0)                  # normalize data for each row
                    axs[i,j].bar(df_scen['Year'], df_scen_mod['NG'], color='r', label='NG', hatch='------')
                    axs[i,j].bar(df_scen['Year'], df_scen_mod['Solar'], color='y', label='Solar', hatch='......', bottom=df_scen_mod['NG'])
                    axs[i,j].bar(df_scen['Year'], df_scen_mod['Wind'], color='b', label='Wind', hatch='//////', bottom=df_scen_mod['NG']+df_scen_mod['Solar'])

                    # axs[i,j].plot(df_scen['Year'], df_scen['NG']/1000, color='r', label="NG", marker=5, markersize=5, markevery=4)
                    # axs[i,j].plot(df_scen['Year'], df_scen['Solar']/1000, color='y', label="Solar", marker='.', markersize=5, markevery=4)
                    # axs[i,j].plot(df_scen['Year'], df_scen['Wind']/1000, color='b', label="Wind", marker='x', markersize=5, markevery=4)
                scen = os.path.basename(scens[s])                                       # get SA scenario
                if load_src=='ERCOT':
                    axs[i,j].set_title('Cost:-'+scen[6:9]+'%, CF:+'+scen[-6:-3]+'%', size=6)    # get subplot title from scenario
                elif load_src=='GCAM':
                    axs[i,j].set_title('Cost:-'+scen[5:8]+'%, CF:+'+scen[-6:-3]+'%', size=6)    # get subplot title from scenario
    
                axs[i,j].set_xticks([2021,2035,2050])
                axs[i,j].set_xticklabels([2021,2035,2050], rotation=90, fontsize=8)
                if plt_type=='cum':
                    axs[i,j].set_yticks(np.arange(0,300,100))
                    axs[i,j].set_yticklabels(np.arange(0,300,100), rotation=0,
                                             fontsize=6)
                elif plt_type=='delta':
                    axs[i,j].set_yticks(np.arange(0,8.1,4))
                    axs[i,j].set_yticklabels(np.arange(0,8.1,4), rotation=0, fontsize=6)
                elif plt_type=='delta%':
                    axs[i,j].set_yticks(np.arange(0,1.1,0.5))
                    axs[i,j].set_yticklabels(np.arange(0,1.1,0.5), rotation=0, fontsize=6)
                axs[i,j].label_outer()                                                  # hide x labels and tick labels for top plots and y ticks for right plots
                s += 1
        fig.supxlabel('Year')
        if plt_type=='cum':
            fig.supylabel('Total generation capacity (GW)')
        elif plt_type=='delta':
            fig.supylabel('Annual investment capacity (GW)')
        elif plt_type=='delta%':
            fig.supylabel('Total Capacity (%)')
        # if load_src=='ERCOT':
        #     fig.suptitle('Annual capacity based on {} load\n Load Scenario: {}'.\
        #               format(load_src, os.path.basename(load_scens[l])))
        # elif load_src=='GCAM':
        #     fig.suptitle('Annual capacity based on {} load\n Load Scenario: {}'.\
        #                   format(load_src, gcam_loads[l]))
        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.75,0.185), ncol=3, 
                    borderaxespad=3, fontsize=8)                                        # one legend for all subplots
        fig.tight_layout()
        plt.show()
        
        
        fig.savefig(os.path.join(output_dir, '01_SA_LoadScen', load_src + '_total_' +
                                 plt_type + os.path.basename(load_scens[l]) + '.png'), 
                    dpi=300, bbox_inches='tight')

### output a mini-table to compare the effect of changes in cost/capacity factor on different technologies:
df_effect = pd.DataFrame(index=[os.path.basename(X) for X in scens], columns=['NG','Solar','Wind'])
for scen in scens:
    df = pd.read_csv(os.path.join(scen, 'Baseline', 'Technology by Year.csv'))
    df_effect['NG'][os.path.basename(scen)] = df['NG'].sum() + 99.8
    df_effect['Solar'][os.path.basename(scen)] = df['Solar'].sum() + 3.9
    df_effect['Wind'][os.path.basename(scen)] = df['Wind'].sum() + 25.1
df_effect.index = ['Cost:-'+os.path.basename(X)[6:9]+'%, CF:+'+os.path.basename(X)[-6:-3]+'%' for X in scens]
df_effect.to_csv(os.path.join(output_dir,'01_SA_LoadScen', 'Effects_table.csv'))


#%% 01_SA_Load - Combined ERCOT and GCAM
"""investments for ERCOT/GCAM in 9 SA scenarios and n load scenarios. 
there is one plot for each load scenario."""
scen_dirs = [sa_gcam_dir, sa_ercot_dir]
load_srcs = ['GCAM', 'ERCOT']
gcam_loads = ['Low Emission, High Population & GDP',
              'Low Emission, Low Population & GDP',
              'Reference Emission, High Population & GDP',
              'Reference Emission, Low Population & GDP']
# cap_initial = {'NG':99849, 'Solar':3975, 'Wind':25123}                                  # initial capacities
result_dir = os.path.join(output_dir, '01_SA_LoadScen')                                 # create result folder if not already there
if os.path.exists(result_dir) == False:
    os.mkdir(result_dir)

plt_type = 'delta'
df_out_g, scens_g, load_scens_g = extract_tech(scen_dirs[0])
df_load_g = df_out_g[df_out_g['Load_Scen'] == os.path.basename(load_scens_g[0])]        # filter data for load scenario
df_out_e, scens_e, load_scens_e = extract_tech(scen_dirs[1])
df_load_e = df_out_e[df_out_e['Load_Scen'] == os.path.basename(load_scens_e[0])]

fig, axs = plt.subplots(3,3)                                                        # plot 9 subplots for 9 SA scenarios
plt.setp(axs, ylim=(0,5800))                                                        # set ylimit for all subplots
s = 0
for i in range(3):                                                              # loop over cost values
    for j in range(3):                                                          # loop over cf values
        df_scen_g = df_load_g[df_load_g['Scen'] == os.path.basename(scens_g[s])]        # filter data by SA scenario
        df_scen_e = df_load_e[df_load_e['Scen'] == os.path.basename(scens_e[s])]
        axs[i,j].plot(df_scen_e['Year'], df_scen_e['NG'], color='r', marker=".", markevery=5, label="NG-ERCOT")
        axs[i,j].plot(df_scen_e['Year'], df_scen_e['Solar'], color='y', marker=".", markevery=5, label="Solar-ERCOT")
        axs[i,j].plot(df_scen_e['Year'], df_scen_e['Wind'], color='b', marker=".", markevery=5, label="Wind-ERCOT")
        axs[i,j].plot(df_scen_g['Year'], df_scen_g['NG'], color='g', marker="^", markevery=5, label="NG-GCAM")
        axs[i,j].plot(df_scen_g['Year'], df_scen_g['Solar'], color='black', marker="^", markevery=5, label="Solar-GCAM")
        axs[i,j].plot(df_scen_g['Year'], df_scen_g['Wind'], color='pink', marker="^", markevery=5, label="Wind-GCAM")
        scen = os.path.basename(scens_e[s])                                             # get SA scenario
        axs[i,j].set_title('Cost:-'+scen[6:9]+'%, CF:+'+scen[-6:-3]+'%', size=8)        # get subplot title from scenario


        axs[i,j].set_xticks([2020,2035,2050])
        axs[i,j].set_xticklabels([2020,2035,2050], rotation=45, fontsize=8)

        axs[i,j].set_yticks(np.arange(0,5800,2000))
        axs[i,j].set_yticklabels(np.arange(0,5800,2000), rotation=45, fontsize=8)
        axs[i,j].label_outer()                                                  # hide x labels and tick labels for top plots and y ticks for right plots
        s += 1
fig.supxlabel('Year')
fig.supylabel('Capacity (MW)')
# if load_src=='ERCOT':
#     fig.suptitle('Annual capacity based on {} load\n Load Scenario: {}'.\
#               format(load_src, os.path.basename(load_scens[l])))
# elif load_src=='GCAM':
#     fig.suptitle('Annual capacity based on {} load\n Load Scenario: {}'.\
#                   format(load_src, gcam_loads[l]))
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, 
            borderaxespad=3, fontsize=7)                                        # one legend for all subplots
fig.tight_layout()
plt.show()


fig.savefig(os.path.join(output_dir, '01_SA_LoadScen', 'combined.png'), 
            dpi=300, bbox_inches='tight')

#%% 02_SA_Tech: technology for SA
"""investments for GCAM in 9 SA scenarios and n load scenarios. 
there is one plot for each technology. ERCOT results are also added as one load
scenario."""
scen_dir1 = sa_gcam_dir
scen_dir2 = sa_ercot_dir
df_out, scens, load_scens = combine(scen_dir1, scen_dir2)                               # combine results from GCAM and ERCOT

result_dir = os.path.join(output_dir, '02_SA_Tech')                                     # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)

load_src = 'GCAM & ERCOT'
colors = ['g', 'r', 'b', 'y', 'magenta']                                                # colors for plots
labels = ['Low Emission, High Population & GDP',
              'Low Emission, Low Population & GDP',
              'Reference Emission, High Population & GDP',
              'Reference Emission, Low Population & GDP',
              'ERCOT - Baseline']                                                       # labels
for tech in ['NG', 'Solar', 'Wind']:                                                    # loop over technologies
    df_tech = df_out[['Year', tech, 'Load_Scen', 'Scen']]                               # filter data for load technology
    fig, axs = plt.subplots(3,3)                                                        # plot 9 subplots for 9 SA scenarios
    plt.setp(axs, ylim=(0,7000))                                                        # set ylimit for all subplots
    s = 0
    for i in range(3):                                                                  # loop over cost values
        for j in range(3):                                                              # loop over cf values
            df_scen = df_tech[df_tech['Scen'] == os.path.basename(scens[s])[5:]]        # filter data by SA scenario
            for l in range(len(load_scens)):
                load_scen = os.path.basename(load_scens[l])                             # get load scenario
                df_loadscen = df_scen[df_scen['Load_Scen']==load_scen]
                axs[i,j].plot(df_loadscen['Year'], 
                              df_loadscen[tech], 
                              color=colors[l], label=labels[l])
            scen = os.path.basename(scens[s])                                           # get SA scenario
            axs[i,j].set_title('Cost:-'+scen[5:8]+'%, CF:+'+scen[-5:-2]+'%', 
                                    size=9)                                             # get subplot title from scenario
            axs[i,j].set_xticks(np.arange(2020,2060,10))
            axs[i,j].set_xticklabels(np.arange(2020,2060,10), rotation=45, fontsize=7)
            axs[i,j].set_yticks(np.arange(0,7000,3000))
            axs[i,j].set_yticklabels(np.arange(0,7000,3000), rotation=45, fontsize=7)
            axs[i,j].label_outer()                                                      # hide x labels and tick labels for top plots and y ticks for right plots
            s += 1
    fig.supxlabel('Year')
    fig.supylabel('Capacity (MW)')
    fig.suptitle('Investments based on {} load\n Technology: {}'.\
                  format(load_src, tech))
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5,-0.2), loc='lower center', 
               ncol=2,  borderaxespad=3, fontsize=7)                                    # use one legend for all subplots
    fig.tight_layout()
    plt.show()
    
    fig.savefig(os.path.join(output_dir, '02_SA_Tech', load_src + '_' + tech + '.png'), 
                dpi=300, bbox_inches='tight')                                           # save figure

#%% 03_SA_Total: total investment for SA
"""investments for GCAM in 9 SA scenarios and n load scenarios. 
there is one plot for each technology. ERCOT results are also added as one load
scenario."""
scen_dir1 = sa_gcam_dir
scen_dir2 = sa_ercot_dir
df_out, scens, load_scens = combine(scen_dir1, scen_dir2)                               # combine results from GCAM and ERCOT

result_dir = os.path.join(output_dir, '03_SA_Total')                                    # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)

load_src = 'GCAM & ERCOT'
colors = ['g', 'r', 'b', 'y', 'magenta']                                                # colors for plots
labels = ['Low Emission, High Population & GDP',
          'Low Emission, Low Population & GDP',
          'Reference Emission, High Population & GDP',
          'Reference Emission, Low Population & GDP',
          'ERCOT - Baseline']
fig, axs = plt.subplots(3,3)                                                            # plot 9 subplots for 9 SA scenarios
plt.setp(axs, ylim=(0,8000))                                                            # set ylimit for all subplots
s = 0
for i in range(3):                                                                      # loop over cost values
    for j in range(3):                                                                  # loop over cf values
        df_scen = df_out[df_out['Scen'] == os.path.basename(scens[s])[5:]]              # filter data by SA scenario
        for l in range(len(load_scens)):
            load_scen = os.path.basename(load_scens[l])                                 # get load scenario
            df_loadscen = df_scen[df_scen['Load_Scen']==load_scen]
            axs[i,j].plot(df_loadscen['Year'], 
                          df_loadscen['Total'], 
                          color=colors[l], label=labels[l])
        scen = os.path.basename(scens[s])                                               # get SA scenario
        axs[i,j].set_title('Cost:-'+scen[5:8]+'%, CF:+'+scen[-5:-2]+'%', 
                                size=9)                                                 # get subplot title from scenario
        axs[i,j].set_xticks(np.arange(2020,2060,10))
        axs[i,j].set_xticklabels(np.arange(2020,2060,10), rotation=45)
        axs[i,j].set_yticks(np.arange(0,8000,2000))
        axs[i,j].set_yticklabels(np.arange(0,8000,2000), rotation=45)
        axs[i,j].label_outer()                                                          # hide x labels and tick labels for top plots and y ticks for right plots
        s += 1
fig.supxlabel('Year')
fig.supylabel('Capacity (MW)')
fig.suptitle('Investments based on {} load'.format(load_src))
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.5,-0.2), loc='lower center', 
           ncol=2, borderaxespad=3, fontsize=7)                                         # one legend for all subplots
fig.tight_layout()
plt.show()

fig.savefig(os.path.join(output_dir, '03_SA_Total', load_src  + '.png'), 
            dpi=300, bbox_inches='tight')

#%% 04_SA_LZ: loadzone for SA
"""investments for ERCOT/GCAM in 9 SA scenarios and n load scenarios. 
there is one plot for each load scenario and technology."""
scen_dirs = [sa_gcam_dir, sa_ercot_dir]
load_srcs = ['GCAM', 'ERCOT']
gcam_loads = ['Low Emission, High Population & GDP',
              'Low Emission, Low Population & GDP',
              'Reference Emission, High Population & GDP',
              'Reference Emission, Low Population & GDP']
LZ_labels = ['AEN','CPS','HOUSTON','NORTH','SOUTH','WEST']                                  # load zones labels
LZ = ['LZ_AEN','LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST']                       # load zones 

colors = ['g', 'r', 'b', 'y', 'm', 'c']                                                     # colors for plots

result_dir = os.path.join(output_dir, '04_SA_LZ')                                           # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)

for b in range(2):
    df_out, scens, load_scens = extract_lz(scen_dirs[b])                                            # generate data for plots
    for ls in range(len(load_scens)):                                                           # loop over load scenarios
        for tech in ['NG', 'Solar', 'Wind']:
            df_load = df_out[df_out['Load_Scen'] == os.path.basename(load_scens[ls])]           # filter data for load scenario
            fig, axs = plt.subplots(3,3)                                                        # plot 9 subplots for 9 SA scenarios
            if tech=='Solar':
                plt.setp(axs, ylim=(0,1000))
            else:
                plt.setp(axs, ylim=(0,5000))                                                        # set ylimit for all subplots
            s = 0
            for i in range(3):                                                                  # loop over cost values
                for j in range(3):                                                              # loop over cf values
                    df_scen = df_load[df_load['Scen'] == os.path.basename(scens[s])]            # filter data by SA scenario
                    for l in range(len(LZ)):
                        axs[i,j].plot(df_scen['Year'], df_scen[LZ[l]+'_'+tech], 
                                      color=colors[l], label=LZ_labels[l])
                    scen = os.path.basename(scens[s])                                           # get SA scenario
                    if load_srcs[b]=='ERCOT':
                        axs[i,j].set_title('Cost:-'+scen[6:10]+'%, CF:+'+scen[-6:-2]+'%', 
                                                size=9)                                         # get subplot title from scenario
                    elif load_srcs[b]=='GCAM':
                        axs[i,j].set_title('Cost:-'+scen[5:9]+'%, CF:+'+scen[-6:-2]+'%', 
                                                size=9)                                         # get subplot title from scenario                                                         
                    axs[i,j].set_xticks(np.arange(2020,2060,10))
                    axs[i,j].set_xticklabels(np.arange(2020,2060,10), rotation=45)
                    
                    if tech=='Solar':
                        axs[i,j].set_yticks(np.arange(0,1000,500))
                        axs[i,j].set_yticklabels(np.arange(0,1000,500), rotation=45)
                    else:
                        axs[i,j].set_yticks(np.arange(0,5000,2000))
                        axs[i,j].set_yticklabels(np.arange(0,5000,2000), rotation=45)
                    axs[i,j].label_outer()                                                      # hide x labels and tick labels for top plots and y ticks for right plots
                    s += 1
            fig.supxlabel('Year')
            fig.supylabel('Capacity (MW)')
            
            if load_srcs[b]=='ERCOT':
                fig.suptitle('{} investments in load zones based on {} load\n Load Scenario: Baseline'.\
                              format(tech, load_srcs[b]))
            elif load_srcs[b]=='GCAM':
                fig.suptitle('{} investments in load zones based on {} load\n Load Scenario: {}'.\
                              format(tech, load_srcs[b], gcam_loads[ls]))
            handles, labels = axs[0,0].get_legend_handles_labels()
            # fig.legend(handles, labels, bbox_to_anchor=(0.5,-0.2), loc='lower center', 
            #            ncol=3, borderaxespad=3, fontsize=7)                                     # one legend for all subplots
            fig.legend(handles, labels, loc='lower center', ncol=6, 
                        borderaxespad=3, fontsize=7)                                        # one legend for all subplots
            fig.tight_layout()
            plt.show()
            
            fig.savefig(os.path.join(output_dir, '04_SA_LZ', load_srcs[b] + '_' +
                                      os.path.basename(load_scens[ls]) + '_' + tech + '.png'), 
                        dpi=300, bbox_inches='tight')

#%% 18_SA_LZ_Total: loadzone for SA (TOTAL values)
"""investments for ERCOT/GCAM in 9 SA scenarios and n load scenarios. 
there is one plot for each load scenario and technology."""
scen_dirs = [sa_gcam_dir, sa_ercot_dir]
load_srcs = ['GCAM', 'ERCOT']
gcam_loads = ['Low Emission, High Population & GDP',
              'Low Emission, Low Population & GDP',
              'Reference Emission, High Population & GDP',
              'Reference Emission, Low Population & GDP']
LZ_labels = ['AEN','CPS','HOUSTON','NORTH','SOUTH','WEST']                                  # load zones labels
LZ = ['LZ_AEN','LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST']                       # load zones 

colors = ['g', 'r', 'b', 'y', 'm', 'c']                                                     # colors for plots

result_dir = os.path.join(output_dir, '18_SA_LZ_total')                                           # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)


for b in range(2):
    df_out, scens, load_scens = extract_lz(scen_dirs[b])                                            # generate data for plots
    for ls in range(len(load_scens)):                                                           # loop over load scenarios
        # for tech in ['NG', 'Solar', 'Wind']:
            df_load = df_out[df_out['Load_Scen'] == os.path.basename(load_scens[ls])]           # filter data for load scenario
            fig, axs = plt.subplots(3,3)                                                        # plot 9 subplots for 9 SA scenarios
            plt.setp(axs, ylim=(0,60))
            s = 0
            for i in range(3):                                                                  # loop over cost values
                for j in range(3):                                                              # loop over cf values
                    df_scen = df_load[df_load['Scen'] == os.path.basename(scens[s])]            # filter data by SA scenario
                    df_scen = df_scen.sum().drop(['Scen', 'Load_Scen', 'Year'])                 # modify df_scen and remove redundant values
                    df_loads = pd.DataFrame(df_scen.values.reshape(6,3), columns=['NG', 'Solar', 'Wind'], index=LZ)
                    # draw bar plots for each load zone
                    axs[i,j].bar(LZ_labels, df_loads['NG']/1000, color='r', label='NG', hatch='------')
                    axs[i,j].bar(LZ_labels, df_loads['Solar']/1000, color='y', 
                                 label='Solar', bottom=df_loads['NG']/1000, hatch='......')
                    axs[i,j].bar(LZ_labels, df_loads['Wind']/1000, color='b', 
                                 label='Wind', bottom=df_loads['NG']/1000+df_loads['Solar']/1000, hatch='//////')
                        
                    scen = os.path.basename(scens[s])                                           # get SA scenario
                    if load_srcs[b]=='ERCOT':
                        axs[i,j].set_title('Cost:-'+scen[6:9]+'% & CF:+'+scen[-6:-3]+'%', 
                                                size=8)                                         # get subplot title from scenario
                    elif load_srcs[b]=='GCAM':
                        axs[i,j].set_title('Cost:-'+scen[5:8]+'% & CF:+'+scen[-6:-3]+'%', 
                                                size=8)                                         # get subplot title from scenario                                                         
                    axs[i,j].set_xticks(LZ_labels)
                    axs[i,j].set_xticklabels(LZ_labels, rotation=45, size=7)
                    axs[i,j].set_yticks(np.arange(0,75,30))
                    axs[i,j].set_yticklabels(np.arange(0,75,30), size=7)
                    
                    s += 1
            # fig.supxlabel('Load zone')
            fig.supylabel('Total investment capacity (GW)')
            
            # if load_srcs[b]=='ERCOT':
            #     fig.suptitle('{} investments in load zones based on {} load\n Load Scenario: Baseline'.\
            #                   format(tech, load_srcs[b]))
            # elif load_srcs[b]=='GCAM':
            #     fig.suptitle('{} investments in load zones based on {} load\n Load Scenario: {}'.\
            #                   format(tech, load_srcs[b], gcam_loads[ls]))
            handles, labels = axs[0,0].get_legend_handles_labels()
            # fig.legend(handles, labels, bbox_to_anchor=(0.5,-0.2), loc='lower center', 
            #            ncol=3, borderaxespad=3, fontsize=7)                                     # one legend for all subplots
            fig.legend(handles, labels, bbox_to_anchor=(0.5,-0.08), loc='lower center', ncol=6, 
                        borderaxespad=3, fontsize=7)                                        # one legend for all subplots
            fig.tight_layout()
            plt.show()
            
            fig.savefig(os.path.join(output_dir, '18_SA_LZ_total', load_srcs[b] + '_' +
                                      os.path.basename(load_scens[ls]) + '.png'), 
                        dpi=300, bbox_inches='tight')


#%% 05_CC_LoadScen: load scenario for CC
"""investments for ERCOT/GCAM in 9 SA scenarios and n load scenarios. 
there is one plot for each load scenario."""
scen_dir = cc_ercot_dir_33
load_src = 'ERCOT'
gcam_loads = ['Low Emission, High Population & GDP',
              'Low Emission, Low Population & GDP',
              'Reference Emission, High Population & GDP',
              'Reference Emission, Low Population & GDP']
df_out, scens, load_scens = extract_tech(scen_dir)                                      # generate data for plots

result_dir = os.path.join(output_dir, '05_CC_LoadScen')                                 # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)

for l in range(len(load_scens)):                                                        # loop over load scenarios
    df_load = df_out[df_out['Load_Scen'] == os.path.basename(load_scens[l])]            # filter data for load scenarios 
    fig, axs = plt.subplots(2,3)                                                        # plot 9 subplots for 9 SA scenarios
    plt.setp(axs, ylim=(0,10))                                                        # set ylimit for all subplots
    s = 1                                                                               # plot ΔT=1-6
    for i in range(2):                                                                  # loop over temperature change values
        for j in range(3):
            df_scen = df_load[df_load['Scen'] == os.path.basename(scens[s])]                # filter data by SA scenario
            # axs[i,j].bar(df_scen['Year'], df_scen['NG'].cumsum()/1000, 
            #              color='r', label="NG", hatch="------")
            # axs[i,j].bar(df_scen['Year'], df_scen['Solar'].cumsum()/1000,
            #              color='y', label="Solar", hatch="......",
            #              bottom=df_scen['NG'].cumsum()/1000)
            # axs[i,j].bar(df_scen['Year'], df_scen['Wind'].cumsum()/1000,
            #              color='b', label="Wind", hatch="//////", 
            #              bottom=df_scen['Solar'].cumsum()/1000 + df_scen['NG'].cumsum()/1000)
            
            axs[i,j].bar(df_scen['Year'], df_scen['NG']/1000, color='r', label='NG', hatch='------')
            axs[i,j].bar(df_scen['Year'], df_scen['Solar']/1000, color='y', label='Solar', hatch='......', bottom=df_scen['NG']/1000)
            axs[i,j].bar(df_scen['Year'], df_scen['Wind']/1000, color='b', label='Wind', hatch='//////', bottom=df_scen['NG']/1000+df_scen['Solar']/1000)
            
            scen = os.path.basename(scens[s])                                               # get SA scenario
            axs[i,j].set_title('ΔT=+'+scen[-1]+'℃', size=9)                              # get subplot title from scenario
            axs[i,j].set_xticks([2021,2035,2050])
            axs[i,j].set_xticklabels([2021,2035,2050], rotation=90)
            axs[i,j].set_yticks(np.arange(0,10,4))
            axs[i,j].set_yticklabels(np.arange(0,10,4), rotation=0)
            axs[i,j].label_outer()                                                      # hide x labels and tick labels for top plots and y ticks for right plots
            s += 1
    fig.supxlabel('Year')
    fig.supylabel('Annual investment capacity (GW)')
    # if load_src=='ERCOT':
    #     fig.suptitle('Investments based on {} load\n Load Scenario: {}, {}'.\
    #              format(load_src, os.path.basename(load_scens[l]), scen_dir[-26:-13]))
    # elif load_src=='GCAM':
    #     fig.suptitle('Investments based on {} load\n Load Scenario: {}'.\
    #                  format(load_src, gcam_loads[l]))
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', 
               ncol=3, borderaxespad=3, fontsize=7)                                             # one legend for all subplots
    fig.tight_layout()
    plt.show()
    
    fig.savefig(os.path.join(output_dir, '05_CC_LoadScen', load_src + '_'
                             + os.path.basename(load_scens[l]) + scen_dir[-26:-13] + '.png'), 
                dpi=300, bbox_inches='tight')
    
#%% 06_CC_Tech: technology for CC
"""investments for GCAM in 9 SA scenarios and n load scenarios. 
there is one plot for each technology. This plot is not meant for ERCOT since
there is only one load scenario in ERCOT."""
scen_dir = cc_ercot_dir_00
load_src = 'ERCOT'

colors = ['g', 'r', 'b', 'y']                                                           # colors for plots
gcam_loads = ['Low Emission, High Population & GDP',
              'Low Emission, Low Population & GDP',
              'Reference Emission, High Population & GDP',
              'Reference Emission, Low Population & GDP']
df_out, scens, load_scens = extract_tech(scen_dir)                                      # generate data for plots

result_dir = os.path.join(output_dir, '06_CC_Tech')                                 # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)

for tech in ['NG', 'Solar', 'Wind']:                                                    # loop over technologies
    df_tech = df_out[['Year', tech, 'Load_Scen', 'Scen']]                               # filter data for load technology
    fig, axs = plt.subplots(2,3)                                                        # plot 9 subplots for 9 SA scenarios
    plt.setp(axs, ylim=(0,5000))                                                        # set ylimit for all subplots
    s = 0
    for i in range(2):                                                                  # loop over cost values
        for j in range(3):                                                              # loop over cf values
            df_scen = df_tech[df_tech['Scen'] == os.path.basename(scens[s])]            # filter data by SA scenario
            for l in range(len(load_scens)):
                load_scen = os.path.basename(load_scens[l])                             # get load scenario
                df_loadscen = df_scen[df_scen['Load_Scen']==load_scen]
                if load_src=='GCAM':
                    axs[i,j].plot(df_loadscen['Year'], 
                                  df_loadscen[tech], 
                                  color=colors[l], label=gcam_loads[l])
                elif load_src=='ERCOT':
                    axs[i,j].plot(df_loadscen['Year'], 
                                  df_loadscen[tech], 
                                  color=colors[l], label='Baseline')
            scen = os.path.basename(scens[s])                                           # get SA scenario
            axs[i,j].set_title('ΔT=+'+scen[8:11]+'℃', size=9)                          # get subplot title from scenario
            axs[i,j].set_xticks(np.arange(2020,2060,10))
            axs[i,j].set_xticklabels(np.arange(2020,2060,10), rotation=45)
            axs[i,j].set_yticks(np.arange(0,5000,2000))
            axs[i,j].set_yticklabels(np.arange(0,5000,2000), rotation=45)
            axs[i,j].label_outer()                                                      # hide x labels and tick labels for top plots and y ticks for right plots
            s += 1
    fig.supxlabel('Year')
    fig.supylabel('Capacity (MW)')
    fig.suptitle('Investments based on {} load\n Technology: {}, {}'.\
                 format(load_src, tech, scen_dir[-26:-13]))
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, 
               borderaxespad=3, fontsize=7)                                             # one legend for all subplots
    
    fig.tight_layout()
    plt.show()
    
    fig.savefig(os.path.join(output_dir, '06_CC_Tech', load_src + tech 
                             + scen_dir[-26:-13] + '.png'), 
                dpi=300, bbox_inches='tight')

#%% 07_CC_Total: total investment for CC
"""investments for GCAM in 9 SA scenarios and n load scenarios. 
there is only one plot for total capacity (NG+Solar+Wind). This plot is not 
meant for ERCOT since there is only one load scenario in ERCOT."""
scen_dir = cc_ercot_dir_00
load_src = 'ERCOT'

colors = ['g', 'r', 'b', 'y']                                                           # colors for plots
gcam_loads = ['Low Emission, High Population & GDP',
              'Low Emission, Low Population & GDP',
              'Reference Emission, High Population & GDP',
              'Reference Emission, Low Population & GDP']
df_out, scens, load_scens = extract_tech(scen_dir)                                      # generate data for plots

result_dir = os.path.join(output_dir, '07_CC_Total')                                 # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)

fig, axs = plt.subplots(2,3)                                                        # plot 9 subplots for 9 SA scenarios
plt.setp(axs, ylim=(0,5000))                                                        # set ylimit for all subplots
s = 0
for i in range(2):                                                                  # loop over cost values
    for j in range(3):                                                              # loop over cf values
        df_scen = df_out[df_out['Scen'] == os.path.basename(scens[s])]            # filter data by SA scenario
        for l in range(len(load_scens)):
            load_scen = os.path.basename(load_scens[l])                             # get load scenario
            df_loadscen = df_scen[df_scen['Load_Scen']==load_scen]
            if load_src=='GCAM':
                axs[i,j].plot(df_loadscen['Year'], 
                              df_loadscen[tech], 
                              color=colors[l], label=gcam_loads[l])
            elif load_src=='ERCOT':
                axs[i,j].plot(df_loadscen['Year'], 
                              df_loadscen[tech], 
                              color=colors[l], label='Baseline')
        scen = os.path.basename(scens[s])                                           # get SA scenario
        axs[i,j].set_title('ΔT=+'+scen[8:11]+'℃', size=9)                          # get subplot title from scenario
        axs[i,j].set_xticks(np.arange(2020,2060,10))
        axs[i,j].set_xticklabels(np.arange(2020,2060,10), rotation=45)
        axs[i,j].set_yticks(np.arange(0,5000,2000))
        axs[i,j].set_yticklabels(np.arange(0,5000,2000), rotation=45)
        axs[i,j].label_outer()                                                      # hide x labels and tick labels for top plots and y ticks for right plots
        s += 1
fig.supxlabel('Year')
fig.supylabel('Capacity (MW)')
fig.suptitle('Investments based on {} load\n{}'.format(load_src, scen_dir[-26:-13]))
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, 
           borderaxespad=3, fontsize=7)                                             # one legend for all subplots
fig.tight_layout()
plt.show()

fig.savefig(os.path.join(output_dir, '07_CC_Total', load_src + scen_dir[-26:-13] + '.png'), 
            dpi=300, bbox_inches='tight')

#%% load scenario for SA - Boxplot
"""investments for ERCOT/GCAM in 9 SA scenarios and n load scenarios. 
there is one plot for each load scenario."""
scen_dir = cc_ercot_dir_00
load_src = 'ERCOT'
gcam_loads = ['Low Emission, High Population & GDP',
              'Low Emission, Low Population & GDP',
              'Reference Emission, High Population & GDP',
              'Reference Emission, Low Population & GDP']
df_out, scens, load_scens = extract_tech(scen_dir)                                      # generate data for plots
for l in range(len(load_scens)):                                                        # loop over load scenarios
    df_load = df_out[df_out['Load_Scen'] == os.path.basename(load_scens[l])]            # filter data for load scenario
    fig, axs = plt.subplots(3,3)                                                        # plot 9 subplots for 9 SA scenarios
    plt.setp(axs, ylim=(0,5000))                                                        # set ylimit for all subplots
    s = 0
    for i in range(3):                                                                  # loop over cost values
        for j in range(3):                                                              # loop over cf values
            df_scen = df_load[df_load['Scen'] == os.path.basename(scens[s])]            # filter data by SA scenario
            axs[i,j].bar(df_scen['Year'], df_scen['NG'], color='r', label="NG")
            axs[i,j].bar(df_scen['Year'], df_scen['Solar'], color='y', label="Solar")
            axs[i,j].bar(df_scen['Year'], df_scen['Wind'], color='b', label="Wind")
            scen = os.path.basename(scens[s])                                           # get SA scenario
            if load_src=='ERCOT':
                axs[i,j].set_title('Cost:-'+scen[6:9]+'%, CF:+'+scen[-5:-2]+'%', size=9)    # get subplot title from scenario
            elif load_src=='GCAM':
                axs[i,j].set_title('Cost:-'+scen[5:8]+'%, CF:+'+scen[-5:-2]+'%', size=9)    # get subplot title from scenario

            axs[i,j].set_xticks(np.arange(2020,2060,10))
            axs[i,j].set_xticklabels(np.arange(2020,2060,10), rotation=45)
            axs[i,j].set_yticks(np.arange(0,5000,2000))
            axs[i,j].set_yticklabels(np.arange(0,5000,2000), rotation=45)
            axs[i,j].label_outer()                                                      # hide x labels and tick labels for top plots and y ticks for right plots
            s += 1
    fig.supxlabel('Year')
    fig.supylabel('Capacity (MW)')
    if load_src=='ERCOT':
        fig.suptitle('Investments based on {} load\n Load Scenario: {}'.\
                 format(load_src, os.path.basename(load_scens[l])))
    elif load_src=='GCAM':
        fig.suptitle('Investments based on {} load\n Load Scenario: {}'.\
                     format(load_src, gcam_loads[l]))
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, 
               borderaxespad=3, fontsize=7)                                             # one legend for all subplots
    fig.tight_layout()
    plt.show()
    
    fig.savefig(os.path.join(output_dir, 'SA_LoadScen', load_src + '_SA_tech_' 
                             + os.path.basename(load_scens[l]) + '_bar.png'), 
                dpi=300, bbox_inches='tight')

#%% 08_CC_StressTest: Climate Stress Test
"""investments for ERCOT/GCAM in 9 SA scenarios and n load scenarios. 
there is one plot for each load scenario."""
scen_dir = cc_ercot_dir_33
load_src = 'ERCOT'
gcam_loads = ['Low Emission, High Population & GDP',
              'Low Emission, Low Population & GDP',
              'Reference Emission, High Population & GDP',
              'Reference Emission, Low Population & GDP']
df_out, scens, load_scens = extract_tech(scen_dir)                                      # generate data for plots

result_dir = os.path.join(output_dir, '08_CC_StressTest')                                 # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)

for l in range(len(load_scens)):                                                        # loop over load scenarios
    df_load = df_out[df_out['Load_Scen'] == os.path.basename(load_scens[l])]            # filter data for load scenario
    df_tech = df_load.groupby('Scen').sum()                                             # group data by temperature scenario
    plt.figure()
    plt.scatter(np.arange(0,7), df_tech['NG'].values, color='r', label='NG')
    plt.scatter(np.arange(0,7), df_tech['Solar'].values, color='y', label='Solar')
    plt.scatter(np.arange(0,7), df_tech['Wind'].values, color='b', label='Wind')
    plt.scatter(np.arange(0,7), df_tech['Total'].values, color='black', label='Total')
    plt.xlabel('ΔT (℃)')
    plt.ylabel('Capacity (MW)')
    if load_src=='ERCOT':
        plt.title('Investments based on {} load\n Load Scenario: {}, {}'.\
             format(load_src, os.path.basename(load_scens[l]), scen_dir[-26:-13]))
    elif load_src=='GCAM':
        plt.title('Investments based on {} load\n Load Scenario: {}, {}'.\
             format(load_src, gcam_loads[l], scen_dir[-26:-13]))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_CC_StressTest', 'ST_' + load_src + '_SA_tech_' 
                             + os.path.basename(load_scens[l]) + scen_dir[-26:-13] 
                             + '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
#%% 09_CC_Risk: Risk in Stress Test
"""this section plots risk values in the modeling period (2021-2050) for all 
agents."""
from matplotlib.dates import DateFormatter
date_form = DateFormatter("%Y")
scen_dir = cc_ercot_dir_00
temp_data_dir = r'C:\Projects\ABM_PGMS\ABM\Weather_Generator\BootStrapWG\Climate_Scenarios'
load_src = 'ERCOT'
scens = [f.path for f in os.scandir(scen_dir) if f.is_dir()]                            # get directories of SA/CC scenarios

result_dir = os.path.join(output_dir, '09_CC_Risk')                                     # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)

for i in range(len(scens)):                                                             # loop over SA/CC scenarios
    df_risk = pd.read_csv(os.path.join(scens[i], 'df_risk.csv'))                        # read risk values
    temp_data = pd.read_csv(os.path.join(temp_data_dir, 'deltaT='+str(i)+'.csv'),
                            parse_dates=['DATE'])                                       # read temperature data
    plot = df_risk.T.plot(title="Agents' risk values for ΔT={} ℃".format(i), 
                          xlabel='Year', ylabel='risk', legend=False)                   # plot data
    # plt.colorbar(label="Risk", orientation="vertical")
    
    plt.text(-8, 0.95, 'risk-tolerant', style='italic', bbox={
        'facecolor': 'green', 'alpha': 0.5, 'pad': 2})
    plt.text(-8, 0.12, 'risk-averse', style='italic', bbox={
        'facecolor': 'red', 'alpha': 0.5, 'pad': 2})       
    fig = plot.get_figure()
    fig.savefig(os.path.join(output_dir, '09_CC_Risk', 'Risk_' + load_src + '_' 
                             + os.path.basename(scens[i]) + '_' + scen_dir[-26:-13] 
                             + '.png'), dpi=300, bbox_inches='tight')
    
#%% 09_CC_Risk: Risk in Stress Test (Combined)
"""this section plots risk values in the modeling period (2021-2050) for all 
agents."""
from matplotlib.dates import DateFormatter
date_form = DateFormatter("%Y")
scen_dir = cc_ercot_dir_00
temp_data_dir = r'C:\Projects\ABM_PGMS\ABM\Weather_Generator\BootStrapWG\Climate_Scenarios'
load_src = 'ERCOT'
scens = [f.path for f in os.scandir(scen_dir) if f.is_dir()]                            # get directories of SA/CC scenarios

result_dir = os.path.join(output_dir, '09_CC_Risk')                                     # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)

df_risk_0 = pd.read_csv(os.path.join(scens[0], 'df_risk.csv'))                          # read risk values for ΔT=0
df_risk_6 = pd.read_csv(os.path.join(scens[6], 'df_risk.csv'))                          # read risk values for ΔT=6

ax = df_risk_0.T.plot(xlabel='Year', ylabel='Climate risk perception (r)', color='#808080', 
                      label='ΔT=0℃', legend=False)                                      # plot data
df_risk_6.T.plot(ax=ax, xlabel='Year', ylabel='Climate risk perception (r)', color='#1A1A1A', 
                 label='ΔT=6℃', legend=False)                                                 # plot data
plt.setp(ax, ylim=(0.15,1))

# ax.set_xticks([2021,2030,2040,2050])
# ax.set_xticklabels([2021,2030,2040,2050])

plt.text(-10, 0.95, 'risk-tolerant', style='italic', bbox={
    'facecolor': 'green', 'alpha': 0.5, 'pad': 2})
plt.text(-10, 0.18, 'risk-averse', style='italic', bbox={
    'facecolor': 'red', 'alpha': 0.5, 'pad': 2})

# create manual legend
import matplotlib.patches as mpatches
patch0 = mpatches.Patch(color='#808080', label='ΔT=0℃')
patch6 = mpatches.Patch(color='#1A1A1A', label='ΔT=6℃')
plt.legend(handles=[patch0, patch6], loc='lower right')

fig = ax.get_figure()
fig.savefig(os.path.join(output_dir, '09_CC_Risk', 'Combined_Risk_' + load_src + '_' 
                         + '_' + scen_dir[-26:-13] 
                         + '.png'), dpi=300, bbox_inches='tight')

#%% 09_CC_Risk: Risk in Stress Test (Combined - 2)
"""this section plots risk values in the modeling period (2021-2050) for all 
agents."""
from matplotlib.dates import DateFormatter
date_form = DateFormatter("%Y")
scen_dir = cc_ercot_dir_00
temp_data_dir = r'C:\Projects\ABM_PGMS\ABM\Weather_Generator\BootStrapWG\Climate_Scenarios'
load_src = 'ERCOT'
scens = [f.path for f in os.scandir(scen_dir) if f.is_dir()]                            # get directories of SA/CC scenarios

result_dir = os.path.join(output_dir, '09_CC_Risk')                                     # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)

df_risk_1 = pd.read_csv(os.path.join(scens[1], 'df_risk.csv'))                          # read risk values for ΔT=1
df_risk_2 = pd.read_csv(os.path.join(scens[2], 'df_risk.csv'))                          # read risk values for ΔT=2
df_risk_3 = pd.read_csv(os.path.join(scens[3], 'df_risk.csv'))                          # read risk values for ΔT=3
df_risk_4 = pd.read_csv(os.path.join(scens[4], 'df_risk.csv'))                          # read risk values for ΔT=4
df_risk_5 = pd.read_csv(os.path.join(scens[5], 'df_risk.csv'))                          # read risk values for ΔT=5

ax = df_risk_1.T.plot(xlabel='Year', ylabel='Climate risk perception (r)', color='#B2BEB5', 
                      label='ΔT=1℃', legend=False)                                      # plot data
df_risk_2.T.plot(ax=ax, xlabel='Year', ylabel='Climate risk perception (r)', color='#7393B3', 
                 label='ΔT=2℃', legend=False)
df_risk_3.T.plot(ax=ax, xlabel='Year', ylabel='Climate risk perception (r)', color='#36454F', 
                 label='ΔT=3℃', legend=False)
df_risk_4.T.plot(ax=ax, xlabel='Year', ylabel='Climate risk perception (r)', color='#A9A9A9', 
                 label='ΔT=4℃', legend=False)
df_risk_5.T.plot(ax=ax, xlabel='Year', ylabel='Climate risk perception (r)', color='#1A1A1A', 
                 label='ΔT=5℃', legend=False)                                                 # plot data
plt.setp(ax, ylim=(0.15,1))

# ax.set_xticks([2021,2030,2040,2050])
# ax.set_xticklabels([2021,2030,2040,2050])

plt.text(-10, 0.95, 'risk-tolerant', style='italic', bbox={
    'facecolor': 'green', 'alpha': 0.5, 'pad': 2})
plt.text(-10, 0.18, 'risk-averse', style='italic', bbox={
    'facecolor': 'red', 'alpha': 0.5, 'pad': 2})

# create manual legend
import matplotlib.patches as mpatches
patch1 = mpatches.Patch(color='#B2BEB5', label='ΔT=1℃')
patch2 = mpatches.Patch(color='#7393B3', label='ΔT=2℃')
patch3 = mpatches.Patch(color='#36454F', label='ΔT=3℃')
patch4 = mpatches.Patch(color='#A9A9A9', label='ΔT=4℃')
patch5 = mpatches.Patch(color='#1A1A1A', label='ΔT=5℃')
plt.legend(handles=[patch1,patch2,patch3,patch4,patch5], loc='lower right')

fig = ax.get_figure()
fig.savefig(os.path.join(output_dir, '09_CC_Risk', 'Combined3_Risk_' + load_src + '_' 
                         + '_' + scen_dir[-26:-13] 
                         + '.png'), dpi=300, bbox_inches='tight')

#%% 19_Agent_Investment: NG investments
"""this function extracts agent investments by year for only NG.
input: Total_Agent_Investment.csv"""

result_dir = os.path.join(output_dir, '19_Agent_Investments')                           # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)
load_src = 'ERCOT'
scen_dir = sa_ercot_dir

# LZ = ['LZ_AEN','LZ_CPS','LZ_HOUSTON','LZ_NORTH','LZ_SOUTH','LZ_WEST']               # load zones
# scens = [f.path for f in os.scandir(scen_dir) if f.is_dir()]                            # get directories of SA/CC scenarios
# for scen in scens:                                                                      # loop over SA/CC scenarios
#     load_scens = [f.path for f in os.scandir(scen) if f.is_dir()]                       # get directories of load scenarios
#     for load_scen in load_scens:                                                        # loop over load scenarios
#         df = pd.read_csv(os.path.join(load_scen, 'Total_Agent_Investment.csv'))         # read the main result
#         agg_year = aggregate(df, agg_type='agent')
#         plot = agg_year.T.plot(xlabel='Year', ylabel='Annual investment in natural gas (MW)', legend=False)
#         fig = plot.get_figure()
#         fig.savefig(os.path.join(output_dir, '19_Agent_Investments', 
#                                  os.path.basename(load_scen) + '_' 
#                                  + os.path.basename(scen)
#                                  + '.png'), dpi=300, bbox_inches='tight')

# load_scen = r'C:\Projects\ABM_PGMS\Results\02_SA\ERCOT\Nov-11 10-55\ERCOT_3.00cost_0.00cf\Baseline'
load_scen = r'C:\Projects\ABM_PGMS\Results\02_SA\ERCOT\Apr-12 23-31\ERCOT_1.50cost_1.50cf\Baseline'
df = pd.read_csv(os.path.join(load_scen, 'Total_Agent_Investment.csv'))         # read the main result
agg_year = aggregate(df, agg_type='agent')
fig, axs = plt.subplots()                                                                # create only one subplot
plt.xlim((2021,2050)); plt.ylim((0,200))
for i in range(len(agg_year)):
    if i in [151,158,160]:
        axs.plot(np.arange(2021,2051), agg_year.iloc[i].values, color='orange')
    else:
        axs.plot(np.arange(2021,2051), agg_year.iloc[i].values, color='gray')
axs.set_xlabel('Year')
axs.set_ylabel('Annual investment capacity in natural gas (MW)')
axs.set_xticks([2021,2030,2035,2040,2050])
axs.set_xticklabels([2021,2030,2035,2040,2050], rotation=0)
axs.set_yticks(np.arange(0,170,50))
axs.set_yticklabels(np.arange(0,170,50), rotation=0)
fig.savefig(os.path.join(output_dir, '19_Agent_Investments', 
                         os.path.basename(load_scen) 
                         + '.png'), dpi=300, bbox_inches='tight')

#%% 10_CC_StressTest: Capacity Deficit
"""investments for ERCOT/GCAM in 9 SA scenarios and n load scenarios. 
there is one plot for each load scenario."""
scen_dir = cc_gcam_dir_33
load_src = 'GCAM'
gcam_loads = ['Low Emission, High Population & GDP',
              'Low Emission, Low Population & GDP',
              'Reference Emission, High Population & GDP',
              'Reference Emission, Low Population & GDP']
df_out, scens, load_scens = extract_total(scen_dir)                                     # generate data for plots

result_dir = os.path.join(output_dir, '10_CC_StressTest_Deficit')                       # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)

for l in range(len(load_scens)):                                                        # loop over load scenarios
    df_load = df_out[df_out['Load_Scen'] == os.path.basename(load_scens[l])]            # filter data for load scenario
    df_temp = df_load.groupby('Scen').sum()                                             # group data by temperature scenario
    plt.figure()
    plt.scatter(np.arange(0,len(df_temp)), df_temp['Capacity Deficit'].values, 
                color='r', label='Capacity Deficit')
    # plt.scatter(np.arange(0,7), df_tech['Total Capacity'].values, color='black', label='Total Capacity')
    plt.xlabel('ΔT (℃)')
    plt.ylabel('Capacity (MW)')
    if load_src=='ERCOT':
        plt.title('Total capacity deficit based on {} load for Cost:-{}%, CF:+{}%\n Load Scenario: {}'.\
             format(load_src, scen_dir[-26:-23], scen_dir[-18:-15], os.path.basename(load_scens[l])))
    elif load_src=='GCAM':
        plt.title('Total capacity deficit based on {} load for Cost:-{}%, CF:+{}%\n Load Scenario: {}'.\
             format(load_src, scen_dir[-26:-23], scen_dir[-18:-15], gcam_loads[l]))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_CC_StressTest_Deficit', 'ST_' + load_src + '_SA_tech_' 
                             + os.path.basename(load_scens[l]) + scen_dir[-26:-13] 
                             + '.png'), dpi=300, bbox_inches='tight')
    plt.show()

#%% 12_CC_Tech: technology for CC
"""investments for GCAM in 9 SA scenarios and n load scenarios. 
there is one plot for each technology. ERCOT results are also added as one load
scenario."""
scen_dir = cc_ercot_dir_00
df_out, scens, load_scens = extract_tech(scen_dir)                               # combine results from GCAM and ERCOT

result_dir = os.path.join(output_dir, '12_CC_Tech')                                     # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)
limits_cum = {'NG':25000, 'Solar': 15000, 'Wind': 100000}
limits_delta = {'NG':7000, 'Solar': 3000, 'Wind': 5000}
plt_type = 'cum'
load_src = 'ERCOT'
colors = ['g', 'b', 'cyan', 'pink','magenta', 'y','r']                                                # colors for plots
gcam_labels = ['Low Emission, High Population & GDP',
              'Low Emission, Low Population & GDP',
              'Reference Emission, High Population & GDP',
              'Reference Emission, Low Population & GDP',
              'ERCOT - Baseline']                                                       # labels
for f in range(len(load_scens)):
    load_scen = load_scens[f]
    df_load = df_out[df_out['Load_Scen'] == os.path.basename(load_scen)]
    for tech in ['NG', 'Solar', 'Wind']:                                                    # loop over technologies
        df_tech = df_load[['Year', tech, 'Load_Scen', 'Scen']]                               # filter data for load technology
        fig, axs = plt.subplots()
        # plt.setp(axs, ylim=(0,7000))                                                        # set ylimit for all subplots
        for i in range(len(scens)):
            df_scen = df_tech[df_tech['Scen'] == os.path.basename(scens[i])]
            if plt_type=='cum':
                axs.plot(df_scen['Year'], df_scen[tech].cumsum(), 
                     color=colors[i], label='ΔT={} ℃'.format(i))
            elif plt_type=='delta':
                axs.plot(df_scen['Year'], df_scen[tech], 
                     color=colors[i], label='ΔT={} ℃'.format(i))

        
        
        fig.supxlabel('Year')
        fig.supylabel('Capacity (MW)')
        if load_src=='ERCOT':
            if plt_type=='cum':
                # axs.set_yticks(np.arange(0,limits_cum[tech],20000))
                # axs.set_yticklabels(np.arange(0,limits_cum[tech],20000), rotation=45)
                axs.set_title('Cumulative capacity based on {} load for Cost:-{}%, CF:+{}%\n Load Scenario: {}, Technology:{}'.\
                     format(load_src, scen_dir[-26:-23], scen_dir[-18:-15], os.path.basename(load_scen), tech))
            elif plt_type=='delta':
                # axs.set_yticks(np.arange(0,limits_delta[tech],2000))
                # axs.set_yticklabels(np.arange(0,limits_delta[tech],2000), rotation=45)
                axs.set_title('Annual investments based on {} load for Cost:-{}%, CF:+{}%\n Load Scenario: {}, Technology:{}'.\
                     format(load_src, scen_dir[-26:-23], scen_dir[-18:-15], os.path.basename(load_scen), tech))
        else:
            if plt_type=='cum':
                # axs.set_yticks(np.arange(0,limits_cum[tech],20000))
                # axs.set_yticklabels(np.arange(0,limits_cum[tech],20000), rotation=45)
                axs.set_title('Cumulative capacity based on {} load for Cost:-{}%, CF:+{}%\n Load Scenario: {}, Technology:{}'.\
                     format(load_src, scen_dir[-26:-23], scen_dir[-18:-15], gcam_labels[f], tech))
            elif plt_type=='delta':
                # axs.set_yticks(np.arange(0,limits_delta[tech],2000))
                # axs.set_yticklabels(np.arange(0,limits_delta[tech],2000), rotation=45)
                axs.set_title('Annual investments based on {} load for Cost:-{}%, CF:+{}%\n Load Scenario: {}, Technology:{}'.\
                     format(load_src, scen_dir[-26:-23], scen_dir[-18:-15], gcam_labels[f], tech))
        handles, labels = axs.get_legend_handles_labels()
        # fig.legend(handles, labels, bbox_to_anchor=(0.5,-0.01), loc='lower center', 
        #            ncol=7,  borderaxespad=3, fontsize=7)                                    # use one legend for all subplots
        fig.legend(handles, labels, bbox_to_anchor=(0.98,0.635),
                   ncol=1, fontsize=10)                                    # use one legend for all subplots
        fig.tight_layout()
        plt.show()
        
        if load_src=='ERCOT':
            fig.savefig(os.path.join(output_dir, '12_CC_Tech', plt_type + '_' + 
                                 load_scen[-44:-31] + '_' + load_src + '_' + tech + '.png'), 
                        dpi=300, bbox_inches='tight')                                           # save figure
        else:
            fig.savefig(os.path.join(output_dir, '12_CC_Tech', plt_type + '_' + 
                                 load_scen[40:53] + '_' + load_src + '_' + tech + '_' + 
                                 os.path.basename(load_scen) + '.png'), 
                        dpi=300, bbox_inches='tight')
            
#%% 12_CC_Tech: NG for CC + deficits (combined)
"""investments for GCAM in 9 SA scenarios and n load scenarios. 
there is one plot for each technology. ERCOT results are also added as one load
scenario."""
# import libraries for zoomed in sections
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector 

def mark_inset_mod(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2

scen_dir = cc_ercot_dir_33
df_out, scens, load_scens = extract_tech(scen_dir)                               # combine results from GCAM and ERCOT
df_out2, scens2, load_scens2 = extract_total(scen_dir) 

result_dir = os.path.join(output_dir, '12_CC_Tech')                                     # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)
load_src = 'ERCOT'
colors = ['g', 'b', 'cyan', 'pink','magenta', 'y','r']                                                # colors for plots
for f in range(len(load_scens)):
    load_scen = load_scens[f]
    df_load = df_out[df_out['Load_Scen'] == os.path.basename(load_scen)]
    df_load2 = df_out2[df_out2['Load_Scen'] == os.path.basename(load_scen)]

    # for tech in ['NG', 'Solar', 'Wind']:                                                    # loop over technologies
    tech = 'Wind'
    df_tech = df_load[['Year', tech, 'Load_Scen', 'Scen']]                               # filter data for load technology
    fig, axs = plt.subplots()
    # plt.setp(axs, ylim=(0,7000))                                                        # set ylimit for all subplots
    axins = zoomed_inset_axes(axs, 12, bbox_to_anchor=(1600,350))
    
    for i in range(len(scens)):
        df_scen1 = df_tech[df_tech['Scen'] == os.path.basename(scens[i])]
        axs.plot(df_scen1['Year'], df_scen1[tech].cumsum()/1000, 
             color=colors[i], label='ΔT={} ℃'.format(i))
        df_scen2 = df_load2[df_load2['Scen'] == os.path.basename(scens[i])]
        axs.plot(df_scen2['Year'], df_scen2['Capacity Deficit'].cumsum()/1000, 
                 color=colors[i], marker='.')
        axins.plot(df_scen1['Year'], df_scen1[tech].cumsum()/1000, 
             color=colors[i])
        
    # draw two lines for markers
    axs.plot(df_scen1['Year'], df_scen1[tech].cumsum()/1000, color='r', label='NG Investment')
    axs.plot(df_scen2['Year'], df_scen2['Capacity Deficit'].cumsum()/1000, 
              color='r', marker='.', label='Deficit')
    

    
    axins.set_xlim(2045, 2046)
    if tech=='NG':
        axins.set_ylim(21, 23.2)
        mark_inset(axs, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    elif tech=='Solar':
        axins.set_ylim(15,17.2)
        mark_inset(axs, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    elif tech=='Wind':
        axins.set_ylim(90,92.2)
        # mark_inset_mod(axs, axins, loc1a=4, loc1b=4, loc2a=1, loc2b=1, fc="none", ec="0.5") 
        mark_inset(axs, axins, loc1=1, loc2=2, ec="0.5")

    axins.set_xticks([])
    axins.set_xticklabels([])
    plt.yticks([])
    mark_inset(axs, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    axs.set_xticks(np.arange(2020,2060,10))
    axs.set_xticklabels(np.arange(2020,2060,10), rotation=0, size=7)
    axs.set_yticks(np.arange(0, 250, 70))
    axs.set_yticklabels(np.arange(0, 250, 70), rotation=0, size=7)
     
    fig.supxlabel('Year')
    fig.supylabel('Capacity (GW)')
    
    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.35,0.88),
                ncol=1, fontsize=8)                                                         # use one legend for all subplots
    fig.legend
    plt.show()
    
    if load_src=='ERCOT':
        fig.savefig(os.path.join(output_dir, '12_CC_Tech', 'Combined3_' + 
                             load_scen[-44:-31] + '_' + load_src + '_' + tech + '.png'), 
                    dpi=300, bbox_inches='tight')                                           # save figure            
        
#%% 13_CC_Deficit: technology for CC
"""investments for GCAM in 9 SA scenarios and n load scenarios. 
there is one plot for each technology. ERCOT results are also added as one load
scenario."""
scen_dir = cc_ercot_dir_33
df_out, scens, load_scens = extract_total(scen_dir)                               # combine results from GCAM and ERCOT

result_dir = os.path.join(output_dir, '13_CC_Deficit')                                     # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)
limits = {'cum':250, 'delta':16}
plt_type = 'cum'
load_src = 'ERCOT'
colors = ['g', 'b', 'cyan', 'pink','magenta', 'y','r']                                                # colors for plots
gcam_labels = ['Low Emission, High Population & GDP',
              'Low Emission, Low Population & GDP',
              'Reference Emission, High Population & GDP',
              'Reference Emission, Low Population & GDP',
              'ERCOT - Baseline']                                                       # labels


for f in range(len(load_scens)):
    load_scen = load_scens[f]
    df_load = df_out[df_out['Load_Scen'] == os.path.basename(load_scen)]
    fig, axs = plt.subplots()
    plt.setp(axs, xlim=(2021,2050), ylim=(0,limits[plt_type]))    
    for i in range(len(scens)):
        df_scen = df_load[df_load['Scen'] == os.path.basename(scens[i])]
        if plt_type=='cum':
            axs.plot(df_scen['Year'], df_scen['Capacity Deficit'].cumsum()/1000, 
                     color=colors[i], label='ΔT={} ℃'.format(i))
        elif plt_type=='delta':
            axs.plot(df_scen['Year'], df_scen['Capacity Deficit']/1000, 
                     color=colors[i], label='ΔT={} ℃'.format(i))
    axs.set_xticks([2021,2035,2050])
    axs.set_xticklabels([2021,2035,2050], rotation=90)
    axs.set_yticks(np.arange(0,limits[plt_type],50))
    axs.set_yticklabels(np.arange(0,limits[plt_type],50), rotation=0)
    fig.supxlabel('Year')
    fig.supylabel('Total capacity deficit (GW)')
    # if load_src=='ERCOT':
    #     axs.set_title('Total capacity deficit based on {} load for Cost:{}%, CF:{}%\n Load Scenario: {}'.\
    #                   format(load_src, scen_dir[-26:-23], scen_dir[-18:-15], os.path.basename(load_scen)))
    # else:
    #     axs.set_title('Total capacity deficit based on {} load for Cost:{}%, CF:{}%\n Load Scenario: {}'.\
    #                   format(load_src, scen_dir[-26:-23], scen_dir[-18:-15], gcam_labels[f]))
    handles, labels = axs.get_legend_handles_labels()
    # fig.legend(handles, labels, bbox_to_anchor=(0.5,-0.01), loc='lower center', 
    #            ncol=7,  borderaxespad=3, fontsize=7)                                    # use one legend for all subplots
    fig.legend(handles, labels, bbox_to_anchor=(0.97,0.635),
               ncol=1, fontsize=10)                                    # use one legend for all subplots
    fig.tight_layout()
    plt.show()
    
    if load_src=='ERCOT':        
        fig.savefig(os.path.join(output_dir, '13_CC_Deficit', plt_type + '_' + 
                                 load_src + '_'  + load_scen[-44:-31] + '.png'), 
                        dpi=300, bbox_inches='tight')                                           # save figure
    else:
        fig.savefig(os.path.join(output_dir, '13_CC_Deficit', plt_type + '_' + 
                                 load_src + '_'  + load_scen[40:53] + '_' + 
                                 os.path.basename(load_scen) + '.png'), 
                        dpi=300, bbox_inches='tight')                                           # save figure

#%% 21_SA_Price_old
"""Price values of energy are extracted from all scenarios and load profiles
and plotted."""
scen_dir = sa_ercot_dir
df_out, scens, load_scens = extract_price(scen_dir)
load_src = 'ERCOT'

result_dir = os.path.join(output_dir, '21_SA_Price')                                     # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)
colors = plt.cm.tab20.colors + plt.cm.tab10.colors[:5]
# colors = ['r', 'g', 'b', 'black', 'cyan', 'magenta', 'y', 'brown', 'pink']
labels = ['Low Emission, High Population & GDP',
              'Low Emission, Low Population & GDP',
              'Reference Emission, High Population & GDP',
              'Reference Emission, Low Population & GDP']

for i in range(len(load_scens)):
    fig, axs = plt.subplots()
    plt.setp(axs, ylim=(10,28), xlim=(2021,2050))
    df_load = df_out[df_out['Load_Scen'] == os.path.basename(load_scens[i])]
    for s in range(len(scens)):
        df_scen = df_load[df_load['Scen'] == os.path.basename(scens[s])]
        axs.plot(df_scen['Year'].astype(float), df_scen['PAVG'], color = colors[s], 
                 label='Cost:-{}% & CF:+{}%'.format(os.path.basename(scens[s])[-15:-12],
                                                   os.path.basename(scens[s])[-6:-3]))

    axs.set_xlabel('Year')
    axs.set_ylabel('Average electricity price in the grid ($/MWh)')
    # axs.set_title('Price values based on {} load\nLoad Scenario: {}'.format(load_src, labels[i]))
    axs.set_xticks([2021,2030,2040,2050])
    axs.set_xticklabels([2021,2030,2040,2050], rotation=0)
    axs.set_yticks(np.arange(10,30,5))
    axs.set_yticklabels(np.arange(10,30,5), rotation=0)
    fig.legend(bbox_to_anchor=(0.44,0.59), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '21_SA_Price', load_src + '_'
                             + os.path.basename(load_scens[i])  
                             + '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
#%% 21_SA_Price_new
"""Price values of energy are extracted from all scenarios and load profiles
and plotted."""
scen_dir = sa_ercot_dir
df_out, scens, load_scens = extract_price_ls(scen_dir, base_loads_dir)
load_src = 'ERCOT'

result_dir = os.path.join(output_dir, '21_SA_Price')                                     # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)
colors = plt.cm.tab20.colors + plt.cm.tab10.colors[:5]
# colors = ['r', 'g', 'b', 'black', 'cyan', 'magenta', 'y', 'brown', 'pink']
labels = ['Low Emission, High Population & GDP',
              'Low Emission, Low Population & GDP',
              'Reference Emission, High Population & GDP',
              'Reference Emission, Low Population & GDP']

for i in range(len(load_scens)):
    fig, axs = plt.subplots()
    plt.setp(axs, ylim=(10,28), xlim=(2021,2050))
    df_load = df_out[df_out['Load_Scen'] == os.path.basename(load_scens[i])]
    for s in range(len(scens)):
        df_scen = df_load[df_load['Scen'] == os.path.basename(scens[s])]
        axs.plot(df_scen['Year'].astype(float), df_scen['PAVG'], color = colors[s], 
                 label='Cost:-{}%, CF:+{}%'.format(os.path.basename(scens[s])[-15:-12],
                                                   os.path.basename(scens[s])[-6:-3]))

    axs.set_xlabel('Year')
    axs.set_ylabel('Average electricity price in the grid ($/MWh)')
    # axs.set_title('Price values based on {} load\nLoad Scenario: {}'.format(load_src, labels[i]))
    axs.set_xticks([2021,2030,2040,2050])
    axs.set_xticklabels([2021,2030,2040,2050], rotation=0)
    axs.set_yticks(np.arange(10,30,5))
    axs.set_yticklabels(np.arange(10,30,5), rotation=0)
    fig.legend(bbox_to_anchor=(1.2,0.98), fontsize=6.5)                                 # use for 5x5 figure
    # fig.legend(bbox_to_anchor=(0.44,0.59), fontsize=9)                            # use for 3x3 figure
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '21_SA_Price', load_src + '_'
                             + os.path.basename(load_scens[i])  
                             + '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
     
    
 
#%% 22_CC_Price
"""Price values of energy are extracted from all scenarios and load profiles
and plotted."""
scen_dir = cc_ercot_dir_33
df_out, scens, load_scens = extract_price(scen_dir)
load_src = 'ERCOT'

result_dir = os.path.join(output_dir, '22_CC_Price')                                     # create result folder if not already there
if os.path.exists(result_dir)==False:
    os.mkdir(result_dir)
colors = ['g', 'b', 'cyan', 'pink','magenta', 'y','r']                                                # colors for plots
labels = ['Low Emission, High Population & GDP',
              'Low Emission, Low Population & GDP',
              'Reference Emission, High Population & GDP',
              'Reference Emission, Low Population & GDP']

for i in range(len(load_scens)):
    fig, axs = plt.subplots()
    df_load = df_out[df_out['Load_Scen'] == os.path.basename(load_scens[i])]
    for s in range(len(scens)):
        df_scen = df_load[df_load['Scen'] == os.path.basename(scens[s])]
        axs.plot(df_scen['Year'].astype(float), df_scen['PAVG'], color = colors[s], 
                 label='ΔT={} ℃'.format(s))

    axs.set_xlabel('Year')
    axs.set_ylabel('Price ($/MWh)')
    if load_src=='GCAM':
        axs.set_title('Price values based on {} load\nLoad Scenario: {}'.format(load_src, labels[i]))
    elif load_src=='ERCOT':
        axs.set_title('Price values based on {} load for Cost:{}%, CF:{}%\nLoad Scenario: Baseline'
                      .format(load_src, scen_dir[-26:-23], scen_dir[-18:-15]))       
    axs.set_xticks(np.arange(2021,2060,10))
    axs.set_xticklabels(np.arange(2021,2060,10), rotation=45)
    axs.set_yticks(np.arange(10,30,5))
    axs.set_yticklabels(np.arange(10,30,5), rotation=45)
    fig.legend(bbox_to_anchor=(0.29,0.555), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, '22_CC_Price', load_src + '_'
                             + os.path.basename(load_scens[i]) + os.path.basename(scen_dir)  
                             + '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    
#%% 50_Capacity_Comparison
"""a- cumulative capacity CURRENT.TRENDS.HIST vs Cost:0.0%, CF:0.0% from ERCOT"""
abm_dir = r'C:\Projects\ABM_PGMS\Results\02_SA\ERCOT\Nov-11 10-55\ERCOT_0.00cost_3.00cf\Baseline'
# abm_dir = r'C:\Projects\ABM_PGMS\Results\02_SA\GCAM\Nov-15 09-52\GCAM_0.00cost_0.00cf\Low Emission-High Population&GDP'
abm_load = 'Lu et al. (2023)'
rds_file = r'C:\Projects\ABM_PGMS\Results\10_Plots\ReEDS Results.xls'                   # ReEDS results
rds_scens = ['CURRENT.TRENDS.HIST',
             'EMERGING.TECH.HIST',
             'HIGH.GROWTH.HIST',
             'HIGH.RE.PENETRATION.HIST']

df_abm = pd.read_csv(os.path.join(abm_dir, 'Technology by Year.csv'))                   # read result file
df_abm = df_abm.iloc[0:14,:]                                                            # filter data for 2021-2034
for r in range(4):                                                                      # loop over ReEDS results
    df_rds = pd.read_excel(rds_file, sheet_name=r)
    
    result_dir = os.path.join(output_dir, '50_Capacity_Comparison')                     # create result folder if not already there
    if os.path.exists(result_dir) == False:
        os.mkdir(result_dir)
        
    cap_initial = {'NG':99.849, 'Solar':3.975, 'Wind':25.123}   
    fig, axs = plt.subplots(1,2)                                                        # 2 subplots
    plt.setp(axs, ylim=(0,210))
    
    # ABM-PGMS
    axs[0].bar(df_abm['Year'], df_abm['NG'].cumsum()/1000 + cap_initial['NG'], 
                 color='r', label="NG")
    axs[0].bar(df_abm['Year'], df_abm['Solar'].cumsum()/1000 + cap_initial['Solar'],
                 color='y', label="Solar", bottom=df_abm['NG'].cumsum()/1000 + cap_initial['NG'])
    axs[0].bar(df_abm['Year'], df_abm['Wind'].cumsum()/1000 + cap_initial['Wind'],
                 color='b', label="Wind", 
                 bottom=df_abm['Solar'].cumsum()/1000 + cap_initial['Solar'] + 
                 df_abm['NG'].cumsum()/1000 + cap_initial['NG'])
    axs[0].set_title('ABM-TX123BT\nCost:0.0%, CF:+3.0%, {} Loads'.format(abm_load))
    
    # ReEDS
    axs[1].bar(df_rds['Year'], df_rds['Gas'], 
                 color='r', label="NG")
    axs[1].bar(df_rds['Year'], df_rds['Solar'],
                 color='y', label="Solar", bottom=df_rds['Gas'])
    axs[1].bar(df_rds['Year'], df_rds['Wind'],
                 color='b', label="Wind", 
                 bottom=df_rds['Solar'] + df_rds['Gas'])
    axs[1].set_title('ReEDS\n{}'.format(rds_scens[r]))
    
    # further plot details
    for i in range(2):
        axs[i].set_xticks([2020,2027,2034])
        axs[i].set_xticklabels([2020,2027,2034], fontsize=7)
        axs[i].set_yticks(np.arange(0,210, 50))
        axs[i].set_yticklabels(np.arange(0,210, 50), fontsize=7)
        axs[i].label_outer()                                                            # hide x labels and tick labels for top plots and y ticks for right plots
    
    fig.supxlabel('Year')
    fig.supylabel('Total Capacity (GW)')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, 
               borderaxespad=3, fontsize=7)                                             # one legend for all subplots
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(output_dir, '50_Comparisons', abm_load + '_' + str(r) 
                         + '.png'), 
            dpi=300, bbox_inches='tight')

#%% 51_Capacity_Comparison
"""a- cumulative capacity ReEDS vs Cost:0.0%, CF:+3.0% from ERCOT"""
abm_dir = r'C:\Projects\ABM_PGMS\Results\02_SA\ERCOT\Apr-12 23-31 - PDM\ERCOT_0.00cost_1.50cf\Baseline'
abm_load = 'Lu & Li (2023)'
rds_file = r'C:\Projects\ABM_PGMS\Results\10_Plots\ReEDS Results.xls'                   # ReEDS results
rds_scens = ['CURRENT.TRENDS.HIST',
             'EMERGING.TECH.HIST',
             'HIGH.GROWTH.HIST',
             'HIGH.RE.PENETRATION.HIST']

df_abm = pd.read_csv(os.path.join(abm_dir, 'Technology by Year + Retirement.csv'))      # read result file
df_abm = df_abm.iloc[0:14,:]                                                            # filter data for 2021-2034
for r in range(4):                                                                      # loop over ReEDS results
    df_rds = pd.read_excel(rds_file, sheet_name=r)
    
    result_dir = os.path.join(output_dir, '50_Capacity_Comparison')                     # create result folder if not already there
    if os.path.exists(result_dir) == False:
        os.mkdir(result_dir)
        
    cap_initial = {'NG':99849, 'Solar':3975, 'Wind':25123}   
    fig, axs = plt.subplots(1,2)                                                        # 2 subplots
    plt.setp(axs, ylim=(0,1))
    
    df_abm_mod1 = df_abm[['NG', 'Solar', 'Wind']].cumsum()                               # extract investments and cumsum
    for tech in ['NG', 'Solar', 'Wind']:                                                # add initial capacity
        df_abm_mod1[tech] = df_abm_mod1[tech] + cap_initial[tech]
    df_abm_mod1 = df_abm_mod1/1000                                                        # convert to GW
    df_abm_mod1 = df_abm_mod1.div(df_abm_mod1.sum(axis=1), axis=0)                         # normalize data for each row
            
    # ABM-PGMS
    axs[0].bar(df_abm['Year'], df_abm_mod1['NG'], color='r', label='NG', hatch='------')
    axs[0].bar(df_abm['Year'], df_abm_mod1['Solar'], color='y', label='Solar',
               bottom=df_abm_mod1['NG'], hatch='......')
    axs[0].bar(df_abm['Year'], df_abm_mod1['Wind'], color='b', label='Wind',
               bottom=df_abm_mod1['NG']+df_abm_mod1['Solar'], hatch='//////')
    axs[0].set_title('ABM-PDM\nCost:0.0%, CF:+1.5%, {} Loads'.format(abm_load),
                     size=8)

    
    df_rds_mod1 = df_rds[['Gas', 'Solar', 'Wind']]                                       # extract investments
    df_rds_mod1 = df_rds_mod1.div(df_rds_mod1.sum(axis=1), axis=0)
    
    
    # ReEDS
    axs[1].bar(df_rds['Year'], df_rds_mod1['Gas'], 
                 color='r', label="NG", hatch='------')
    axs[1].bar(df_rds['Year'], df_rds_mod1['Solar'], hatch='......',
                 color='y', label="Solar", bottom=df_rds_mod1['Gas'])
    axs[1].bar(df_rds['Year'], df_rds_mod1['Wind'],
                 color='b', label="Wind", hatch='//////',
                 bottom=df_rds_mod1['Solar'] + df_rds_mod1['Gas'])
    axs[1].set_title('ReEDS\n{}'.format(rds_scens[r]), size=8)
    
    # further plot details
    for i in range(2):
        axs[i].set_xticks([2020,2027,2034])
        axs[i].set_xticklabels([2020,2027,2034], fontsize=7)
        axs[i].set_yticks(np.arange(0,1,0.25))
        axs[i].set_yticklabels(np.arange(0,1,0.25), fontsize=7)
        axs[i].label_outer()                                                            # hide x labels and tick labels for top plots and y ticks for right plots
    
    fig.supxlabel('Year')
    fig.supylabel('Total Capacity (%)')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, 
               borderaxespad=3, fontsize=7.5)                                             # one legend for all subplots
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(output_dir, '50_Capacity_Comparison', abm_load + '_' + str(r) 
                         + '.png'), 
            dpi=300, bbox_inches='tight')



"""b- cumulative capacity ReEDS vs Cost:-3.0%, CF:+3.0% from GCAM"""
abm_dir = r'C:\Projects\ABM_PGMS\Results\02_SA\GCAM\Apr-19 10-05 - PDM\GCAM_3.00cost_3.00cf\Low Emission-High Population&GDP'
abm_load = 'GCAM'
rds_file = r'C:\Projects\ABM_PGMS\Results\10_Plots\ReEDS Results.xls'                   # ReEDS results
rds_scens = ['CURRENT.TRENDS.HIST',
             'EMERGING.TECH.HIST',
             'HIGH.GROWTH.HIST',
             'HIGH.RE.PENETRATION.HIST']

df_abm = pd.read_csv(os.path.join(abm_dir, 'Technology by Year + Retirement.csv'))      # read result file
df_abm = df_abm.iloc[0:14,:]                                                            # filter data for 2021-2034
for r in range(4):                                                                      # loop over ReEDS results
    df_rds = pd.read_excel(rds_file, sheet_name=r)
    
    result_dir = os.path.join(output_dir, '50_Capacity_Comparison')                     # create result folder if not already there
    if os.path.exists(result_dir) == False:
        os.mkdir(result_dir)
        
    cap_initial = {'NG':99849, 'Solar':3975, 'Wind':25123}   
    fig, axs = plt.subplots(1,2)                                                        # 2 subplots
    plt.setp(axs, ylim=(0,1))
    
    df_abm_mod2 = df_abm[['NG', 'Solar', 'Wind']].cumsum()                               # extract investments and cumsum
    for tech in ['NG', 'Solar', 'Wind']:                                                # add initial capacity
        df_abm_mod2[tech] = df_abm_mod2[tech] + cap_initial[tech]
    df_abm_mod2 = df_abm_mod2/1000                                                        # convert to GW
    df_abm_mod2 = df_abm_mod2.div(df_abm_mod2.sum(axis=1), axis=0)                         # normalize data for each row
            
    # ABM-PGMS
    axs[0].bar(df_abm['Year'], df_abm_mod2['NG'], color='r', label='NG', hatch='------')
    axs[0].bar(df_abm['Year'], df_abm_mod2['Solar'], color='y', label='Solar',
               bottom=df_abm_mod2['NG'], hatch='......')
    axs[0].bar(df_abm['Year'], df_abm_mod2['Wind'], color='b', label='Wind',
               bottom=df_abm_mod2['NG']+df_abm_mod2['Solar'], hatch='//////')
    axs[0].set_title('ABM-PDM\nCost:-3.0%, CF:+3.0%, {} Loads'.format(abm_load),
                     size=8)

    
    df_rds_mod2 = df_rds[['Gas', 'Solar', 'Wind']]                                       # extract investments
    df_rds_mod2 = df_rds_mod2.div(df_rds_mod2.sum(axis=1), axis=0)                      # normalize data for each row
    
    
    # ReEDS
    axs[1].bar(df_rds['Year'], df_rds_mod2['Gas'], 
                 color='r', label="NG", hatch='------')
    axs[1].bar(df_rds['Year'], df_rds_mod2['Solar'], hatch='......',
                 color='y', label="Solar", bottom=df_rds_mod2['Gas'])
    axs[1].bar(df_rds['Year'], df_rds_mod2['Wind'],
                 color='b', label="Wind", hatch='//////',
                 bottom=df_rds_mod2['Solar'] + df_rds_mod2['Gas'])
    axs[1].set_title('ReEDS\n{}'.format(rds_scens[r]), size=8)
    
    # further plot details
    for i in range(2):
        axs[i].set_xticks([2020,2027,2034])
        axs[i].set_xticklabels([2020,2027,2034], fontsize=7)
        axs[i].set_yticks(np.arange(0,1,0.25))
        axs[i].set_yticklabels(np.arange(0,1,0.25), fontsize=7)
        axs[i].label_outer()                                                            # hide x labels and tick labels for top plots and y ticks for right plots
    
    fig.supxlabel('Year')
    fig.supylabel('Total Capacity (%)')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, 
               borderaxespad=3, fontsize=7.5)                                             # one legend for all subplots
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(output_dir, '50_Capacity_Comparison', abm_load + '_' + str(r) 
                         + '.png'), 
            dpi=300, bbox_inches='tight')

#%% Investment table
result_dir = os.path.join(output_dir, '54_CC_investment')                                     # create result folder if not already there
if os.path.exists(result_dir) == False:
    os.mkdir(result_dir)

df_ercot = extract_tech_total(cc_ercot_dir_33)
df_gcam = extract_tech_total(cc_gcam_dir_33)


#%% 53_Fadeout table
result_dir = os.path.join(output_dir, '53_Fadeout')                                     # create result folder if not already there
if os.path.exists(result_dir) == False:
    os.mkdir(result_dir)

# input data
wd = r'C:\Projects\ABM_PGMS\Results\03_CC\ERCOT\3.0cost_3.0cf\Apr-17 11-37 - PDM\deltaT=6'    # working directory
rsr_file = r'C:\Projects\ABM_PGMS\ABM\Future_Data\dist_RSR_Nov-10 21-35_9.csv'      # RSR file
df_rsr = pd.read_csv(rsr_file)
deficit_file = os.path.join(wd, 'baseline', 'Total_Annual_Investment.csv')          # deficit file
df_deficit = pd.read_csv(deficit_file, index_col=['Year'])
agt_invest_file = os.path.join(wd, 'baseline', 'Total_Agent_Investment.csv')        # agent investments file
df_agent = pd.read_csv(agt_invest_file)
risk_file = os.path.join(wd, 'df_risk.csv')
df_risk = pd.read_csv(risk_file)

# deficit
df_out_def = pd.DataFrame()
df_test = pd.DataFrame()
for y in np.arange(2021,2051):                                                      # loop over years
    df_out_def[y] = df_rsr['size']*df_deficit.loc[y]['Capacity Deficit']            # annual deficit for each agent
    df_test[y] = df_rsr['size']*df_deficit.loc[y]['Capacity Deficit']*df_risk[str(y)]
# investments
agg_year = aggregate(df_agent, agg_type='agent', tech='NG')                                    # final investments
df_out_invest = pd.DataFrame(agg_year.values/df_risk.values, 
                             columns=np.arange(2021,2051))                          # investments before applying r values

# output results for some agents in one year as a table
agents = [80,105,137]
years = [2021,2030,2040]
for year in years:
    df_out = pd.DataFrame(index=agents,columns=['Deficit','Investment without r', 'Investment with r'])
    for i in agents:
        df_out.loc[i]['Deficit'] = df_out_def.loc[i][year].round(1)
        df_out.loc[i]['Investment without r'] = df_out_invest.loc[i][year].round(1)
        df_out.loc[i]['Investment with r'] = agg_year.loc[i][year].round(1)
        
    df_out.to_csv(os.path.join(result_dir, '{}_fadeout_{}.csv'.
                            format(os.path.basename(wd),year)))                  # save as *.csv
    
#%% 53_Fadeout table for all agents
result_dir = os.path.join(output_dir, '53_Fadeout')                                     # create result folder if not already there
if os.path.exists(result_dir) == False:
    os.mkdir(result_dir)
tech = 'Wind'
for dt in [0,3,6]:
    # input data
    wd = f'C:\\Projects\\ABM_PGMS\\Results\\03_CC\\ERCOT\\3.0cost_3.0cf\\Apr-17 11-37 - PDM\\deltaT={str(dt)}'    # working directory
    rsr_file = r'C:\Projects\ABM_PGMS\ABM\Future_Data\dist_RSR_Nov-10 21-35_9.csv'      # RSR file
    df_rsr = pd.read_csv(rsr_file)
    deficit_file = os.path.join(wd, 'baseline', 'Total_Annual_Investment.csv')          # deficit file
    df_deficit = pd.read_csv(deficit_file, index_col=['Year'])
    agt_invest_file = os.path.join(wd, 'baseline', 'Total_Agent_Investment.csv')        # agent investments file
    df_agent = pd.read_csv(agt_invest_file)
    risk_file = os.path.join(wd, 'df_risk.csv')
    df_risk = pd.read_csv(risk_file)
    
    # deficit
    df_out_def = pd.DataFrame()
    df_test = pd.DataFrame()
    for y in np.arange(2021,2051):                                                      # loop over years
        df_out_def[y] = df_rsr['size']*df_deficit.loc[y]['Capacity Deficit']            # annual deficit for each agent
        df_test[y] = df_rsr['size']*df_deficit.loc[y]['Capacity Deficit']*df_risk[str(y)]
    # investments
    agg_year = aggregate(df_agent, agg_type='agent', tech=tech)                         # final investments
    df_out_invest = pd.DataFrame(agg_year.values/df_risk.values, 
                                 columns=np.arange(2021,2051))                          # investments before applying r values
    
    # output results for some agents in one year as a table
    # agents = [80,105,137]
    agents = np.arange(0,161)
    years = [2021, 2025, 2030, 2035, 2040, 2045]
    for year in years:
        df_out = pd.DataFrame(index=agents,columns=['Deficit','Investment without r', 'Investment with r'])
        for i in agents:
            # df_out.loc[i]['Deficit'] = df_out_def.loc[i][year].round(1)
            # df_out.loc[i]['Investment without r'] = df_out_invest.loc[i][year].round(1)
            # df_out.loc[i]['Investment with r'] = agg_year.loc[i][year].round(1)
            df_out.loc[i]['Deficit'] = df_out_def.loc[i][year].round(2)
            df_out.loc[i]['Investment without r'] = df_out_invest.loc[i][year].round(2)
            df_out.loc[i]['Investment with r'] = agg_year.loc[i][year].round(2)
        ### additional columns for finding similarities between agents' behavior
        df_out['Percentage without r'] = df_out['Investment without r'] / df_out['Deficit']
        df_out['Percentage with r'] = df_out['Investment with r'] / df_out['Deficit']
        df_out['Percentage difference'] = (df_out['Investment without r'] - df_out['Investment with r']) / df_out['Deficit']
        # df_out = df_out.round(decimals=2)
        df_out.to_csv(os.path.join(result_dir, f'{os.path.basename(wd)}_fadeout_{year}_{tech}.csv'))     # save as *.csv
        
#%% 54_ 3d boxplots
result_dir = os.path.join(output_dir, '53_Fadeout')                                     # create result folder if not already there
df_combined = pd.DataFrame()
for year in [2021, 2025, 2030, 2035, 2040, 2045]:
    for dt in [0,3,6]:
        for tech in [None, 'NG', 'Solar', 'Wind']:
            df = pd.read_csv(os.path.join(result_dir, f'deltaT={dt}_fadeout_{year}_{tech}.csv'))
            df_combined[f'{tech}_{dt}_{year}'] = df['Percentage with r']
df_combined.to_csv(os.path.join(result_dir, '3d_hist.csv'))
# plt.setp(ax, ylim=(0,1))                                                 # set ylimit for all subplots

for year in [2021, 2025, 2030, 2035, 2040, 2045]:
    for dt in [0,3,6]:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.axes.set_ylim3d(bottom=0, top=1)
        ax.axes.set_zlim3d(bottom=0, top=1) 

        ax.set_title(f'Year {year}, ΔT={dt} ℃')
        xs = np.arange(161)
        ax.bar(xs, df_combined[f'NG_{dt}_{year}'], zs=0, zdir='y', color='r')
        ax.bar(xs, df_combined[f'Solar_{dt}_{year}'], zs=1, zdir='y', color='y')
        ax.bar(xs, df_combined[f'Wind_{dt}_{year}'], zs=2, zdir='y', color='b')
        ax.bar(xs, df_combined[f'None_{dt}_{year}'], zs=3, zdir='y', color='black')
        
        ax.set_xlabel('Agent', fontsize=12)
        ax.set_ylabel('Technology', fontsize=12)
        ax.set_zlabel('Investment/Deficit (%)', fontsize=12)
        
        ax.xaxis.set_tick_params(size=8)
        ax.zaxis.set_tick_params(size=8)
        ax.set_yticks(range(4), ['NG','Solar','Wind','All'], fontsize=8)
        plt.savefig(os.path.join(result_dir, 'plots', f'{dt}_{year}.png'), 
                dpi=300, bbox_inches='tight')
        plt.show()

#%% 55_ boxplot for each technology (2D)
result_dir = os.path.join(output_dir, '53_Fadeout')                                     # create result folder if not already there
df_combined = pd.DataFrame()
for year in [2021, 2025, 2030, 2035, 2040, 2045]:
    for dt in [0,3,6]:
        for tech in [None, 'NG', 'Solar', 'Wind']:
            df = pd.read_csv(os.path.join(result_dir, f'deltaT={dt}_fadeout_{year}_{tech}.csv'))
            df_combined[f'{tech}_{dt}_{year}'] = df['Percentage with r']
df_combined.to_csv(os.path.join(result_dir, '3d_hist.csv'))
# plt.setp(ax, ylim=(0,1))                                                 # set ylimit for all subplots

years = [2021, 2025, 2030, 2035, 2040, 2045]
for tech in ['NG','Solar','Wind']:
    if tech=='NG': color='r'
    elif tech=='Solar': color='y'
    else: color='b'
    for dt in [0]:
    # for dt in [0,3,6]:
        fig, ax = plt.subplots(1,6, figsize=(12,2), sharex=True, sharey=True)
        plt.setp(ax, ylim=(0,1))                                                 # set ylimit for all subplots
        # ax.set_ylim(bottom=0, top=1)
        # ax.set_title(f'Technology: {tech}, ΔT={dt} ℃')
        xs = np.arange(161)
        ax[0].bar(xs, df_combined[f'{tech}_{dt}_2021'], color=color)
        ax[1].bar(xs, df_combined[f'{tech}_{dt}_2025'], color=color)
        ax[2].bar(xs, df_combined[f'{tech}_{dt}_2030'], color=color)
        ax[3].bar(xs, df_combined[f'{tech}_{dt}_2035'], color=color)
        ax[4].bar(xs, df_combined[f'{tech}_{dt}_2040'], color=color)
        ax[5].bar(xs, df_combined[f'{tech}_{dt}_2045'], color=color)
        
        ax[0].set_ylabel('Investment/Deficit (%)', fontsize=8)
        # ax[1,0].set_ylabel('Investment/Deficit (%)', fontsize=8)
        for i in range(6):
            ax[i].set_xlabel('Agent', fontsize=8)
            ax[i].set_title(f'{years[i]}')
        plt.savefig(os.path.join(result_dir, 'plots', f'{tech}_{dt}.png'), 
                dpi=300, bbox_inches='tight')
        plt.show()