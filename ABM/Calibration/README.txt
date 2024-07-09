1- Calibration.py:
Determines the initial values for cost, risk, REC, and size for all agents based
on dist_params as the input. the code outputs KGE results but the best result is
selected manually based on how close the simulated capacity is to the real 
values across all three technologies. the code determines investment decisions 
for 10 times for each set of dist_params to capture stochasticity and outputs
the results as figures.
look at the generated figures for each input and select the best. use the 
corresponding dist_cost.csv and dist_RSR.csv for future simulations.
results are saved in '/Results/00_Calibration'

2- Historical_Analysis_Stochastic.py
determines the historical behavior of agents for different values of risk given
by predefined values of r as risk_values = [0.2, 0.4, 0.6, 0.8, 1.0, -99]. the 
code is simulated 10 times for each r value to capture stochasticity.
results are saved in '/Results/01_Historical_Analysis'

3- Historical_Analysis_Deterministic.py
is similar to Historical_Analysis_Stochastic.py but does not have a stochastic
component. the code runs once for each r value.
results are saved in '/Results/01_Historical_Analysis'