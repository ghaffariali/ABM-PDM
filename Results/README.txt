Follow this guide for every scenario in each folder.
an example of a path for results:
C:\ABM-PDM\Results\02_SA\ERCOT\Apr-12 23-31 - PDM\ERCOT_0.00cost_0.00cf\Baseline

Agent_Investments: 
	investments for agents by load zone and technology

Bus_Investments:
	agent investments converted to bus level. This is done according to loads at
	bus level.
	
frame:
	raw files to create input files for PGMS
	
Generators:
	list of generators at each year
	
Generators_C:
	list of generators with cumulative capacities. Agent investments are converted
	to bus level and new capacities. These new capacities are added to "existing"
	generators to expand capacity. Every 5 years, new generators are built. So, 
	over time, every 5 years, the number of generators increase.
	
Loads_2Base:
	predicted loads without subtracting renewable capacities
	
Loads_3Updated:
	modified loads after subtracting renewable capacities
	
PGMS_Inputs:
	inputs created for PGMS using Generators_C and Loads_3Updated
	
PGMS_Outputs:
	solutions from PGMS
	
Excel files:
	results at agent level with different spatial resolutions and settings