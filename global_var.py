"""
Author Lilian Bosc 
Latest update: 13/12/2022

Initial variables which will be used in the code. These variables can be modified by the code main.py

commented
"""

import os
import json

# Default global variables
filename = "modelain2.0"

max_row = 164
max_col = 118
gen_wise_data = {}
glo_n_layers = 1

clustering_algo = 'ward' # see wells_clustering.py to know all the possible clustering methods

# MOPSO variables
N_iter = 500
N_pop = 25
archive_size = "auto" # see mopso_tools.py line 421
n_neighboors = 4
w = 0.7
c1 = 1
c2 = 2
mode = "swarm"
p_turbulence = 0.1
penalty = True

all_cost = []
threshold = 4

dthreshold = 2
Cmodel = 2006.7

# How to find a representant for each area
representant_findby="2D-median"

# Discharges range
Qlb = -2500
Qub = -500
# to improve: it can be different in each area

# Initial population range
init_range = [Qlb, Qub]
# initial range should be inclueded in search space
# to improve: it can be different in each area

opti_folder = ""
environment_file_path = "data/"+filename+"_MODFLOW/"+filename+".h5"
heads_file_path = "data/"+filename+"_MODFLOW/"+filename+".hed"
leakage_file_path = "data/"+filename+"_MODFLOW/"+filename+".ccf"
drawdown_file_path = "data/"+filename+"_MODFLOW/"+filename+".drw"
training_file_path = "data/"+filename+"_MODFLOW/training.h5"

output_folder_path = "data/"+opti_folder+"/"+filename+"_output"
output_costs_path = output_folder_path + f"/mopso_costs-{N_iter}-{N_pop}_{filename}.csv"
output_memory_path = output_folder_path + f"/mopso_memory-{N_iter}-{N_pop}_{filename}.txt"
output_pareto_opt_path = output_folder_path + f"/mopso_pareto_opt-{N_iter}-{N_pop}_{filename}.csv"
output_penalties_path = output_folder_path + f"/mopso_penalties_opt-{N_iter}-{N_pop}_{filename}.csv"
clusters_file_path = output_folder_path + "/clusters.csv"