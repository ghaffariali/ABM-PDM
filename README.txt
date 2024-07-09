This is the guide for coupled model ABM-PDM.
First, PDM runs for the year y and determines the price at each bus for the next modeling period.
Taking the price inputs from PDM, ABM determines the future investment decisions at bus level for three
technology types (NG, Solar, Wind). NG capacity is added to the generators list as a new generator and solar
and wind capacity are deducted from load assuming they can satisfy the amount of load equal to their capacity.

######### Model initialization
Required libraries and packages to run the code:
1-	Numpy
2-	Pandas
3-	Numpy-financial
4-	Sklearn
5-	Openpyxl
6-	matplotlib
7-	pyomo
8-	xlsxwriter
9-	pickle
10-	glpk solver

######## Basic:
Run the model for predetermined scenarios for change in future cost, capacity factor, and average temperature

1- run the desired scenario from '/Scenarios': 
	CC is for Climate Change
	SA for Sensitivity Analysis
	MultiScen.py files are for simulating multiple scenarios.
	SingleScen.py files are for simulating only one scenario which can be edited
	through input files.

######## Advanced:
Run the model for custom settings:

1- run Calibration.py in '/Calibration' to initialize the model. all required
data are stored in '/Calibration/input'. select the best results for dist_Cost.csv
and dist_RSR.csv and copy them in '/Future_Data'.
2- run future_data_generator.py with the desired values to generate future data
for CF, Cost, and Retired capacity.
3- run load_preprocess.py to prepare load values for future simulation.
4- run the desired scenario from '/Scenarios': 
	CC is for Climate Change
	SA for Sensitivity Analysis
	MultiScen.py files are for simulating multiple scenarios.
	SingleScen.py files are for simulating only one scenario which can be edited
	through input files.

######## Model parameters in 8.py files in Scenarios folder
T=30: modeling period (years to run the model for). By default model runs for 2021-2050.
ys=1: modeling interval. By default, model runs annually.
n=5: investment interval. By default, new generators are added to the network once every 5 years.
In PDM section, the simplified version of the optimization model runs to determine the prices.

######## Results
Results are stored in "Results" folder in a folder as the date. Folders inside the main result directory:

Agent_Investments: investment results by agent
Bus_Investments: investment results by bus
frame: contains files needed to create pyomo input files
Generators: includes the generators list for each modeling year (only NG).
Generators_C: includes the cummulative generators list every n years (only NG). this list speeds up 
PGMS and also assumes that investments are developed over time.
Loads_2Base: includes vectorized loads in 25 hours and 123 buses in different load profiles.
Loads_3Updated: includes hourly load values after subtracting solar and wind capacities.
PGMS_Inputs: contains pyomo input files for PGMS model
PGMS_Outputs: contains PGMS output files (hourly price, load, and power output at each bus)

Investment results are available at agent and bus level by technology type.
Note: Results in file 'Technology by Year.csv" are the remaining capacities for each technology at the end of the year.
IOW, retired capacity has been subtracted from NG technology.
Also, the results are available in the form of Load Zone by Year, Technology by Agent, and Technology by Year.

######## Python files
01 - functions.py
	contains all classes and functions required to run ABM-PDM.
	
02 - future_data_generator.py
	this code creates future data for 'Cost', 'CF', and 'Retirement' based on 
	annual change rate. results are stored in '/Future_Data' in '/CF', '/Cost', 
	and '/Retirement'
	
03 - load_preprocess.py
	this code preprocesses load input data for GCAM and ERCOT. The input is raw
	load for GCAM and ERCOT in tabular format; the output is annual load for 
	GCAM and ERCOT in vectorized format with hour 0 added. additional info is
	available in the README file in '/Load_Data'.

04 - Plots.py
	this code outputs plots for the ABM-PDM project.
