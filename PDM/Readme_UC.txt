### The power system daily operation model (Unit Commitment / UC)
### Interpreter: Python 3.8
### Packages required: pyomo, pickle, xlsxwriter

Two version:
1) A simplified UC model (less time for simulation)
2) A complete UC model (more time for simulation, but more accurate)

1) The simplified UC model:
Run 'UC_simplified_function.py' file
This file defines several functions, and the sample codes to run simplified UC is at the end of this file.
Input: 'formpyomo_UC_data.dat'  # includes all the power system data required for UC
Output: 'UCsolution.dat'   # solution for the first run of UC
        'UCsolution_run2.dat'   # solution for the second run of UC (The second run can obtain the electricity price)

2) The complete UC model:
Run 'UC_function.py' file
This file defines several functions, and the sample codes to run UC is at the end of this file.
Input: 'formpyomo_UC_data.dat'  # includes all the power system data required for UC
Output: 'UCsolution.dat'   # solution for the first run of UC
        'UCsolution_run2.dat'   # solution for the second run of UC (The second run can obtain the electricity price)

Data file format:
1) 'formpyomo_UC_data.dat':
string: 'param: BUS : bus_num := '
column 1: Bus set index
column 2: bus_number
string: ';'
string: 'param: GEN: gen_num gen_bus gen_cost_P gen_cost_NL gen_cost_SU gen_Pmin gen_Pmax gen_r10 gen_rr := '
column 1: generator set index
column 2: generator number
column 3: generator's bus number
column 4: generator's operational cost ($/MWh)
column 5: generator's no load cost ($/hr)
column 6: generator's start-up cost ($/start-up)
column 7: generator's minimum power when generator is online (MW)
column 8: generator's maximum power when generator is online (MW)
column 9: generator's ramping rate in 10 minutes (MW/10min)
column 10: generator's ramping rate (MW/hr)
string: ';'
string: 'param: LINE: line_num line_x line_Pmax line_fbus line_tbus := '
column 1: line set index
column 2: line number
column 3: line reactance per unit
column 4: line maximum power capacity (MW)
column 5: bus number of the bus at the beginning of the line
column 6: bus number of the bus at the end of the line
string: ';'
string: 'param: TIME: time_num := '
column 1: time set index
column 2: time period number
string: ';'
string: 'param: BUS_TIME: bustime_num load_b_t:= '
column 1: bus set index
column 2: time set index
column 3: bus_time set index
column 4: load at the bus in the time period (MW)
string: ';'

2) 'UCsolution.dat'/'UCsolution_run2.dat'/'UCsolution_full.dat'/'UCsolution_full_run2.dat'
'p_g_t':
row: each generator
column: the generator output power(MW) for each time period (hour)  # include initial hour, hence it has 25 hours
'p_k_t':
row: each line
column: the power(MWh) on transmission line for each time period (hour)
'p_k_t_pct':
row: each line
column: the percentage of the power to the line capacity for each time period (hour)
'r_g_t':
row: each generator
column: the generator reservation (MW) for each time period (hour)

3) 'LMP_UC_simplified.csv'/'LMP_UC_full.csv'
The file that includes the electricity price on each bus.

