"""
Author Lilian Bosc 
Latest update: 13/12/2022

Recover the data from main and run the optimisation by calling mopso_tools or moga_tools.

commented
"""

import global_var
from modules import *
import data_management_tools as dm
from mopso_tools import *
from moga_tools import *

def optimize(optimisation_method, clustering_algo, display=False, blackout=False, save=None):
    """
    This function will launches the optimisation according to the parameters, saves the files and displays 
    the results if required.
    """
    go_to_the_root() # return at the root of the folder
    if not os.path.exists(global_var.output_folder_path):
        os.makedirs(global_var.output_folder_path) # create the folder which will contain the outputs
    
    ## Setting up variables for the model
    wells, areas, river = dm.read_cells_clustering(clustering_algo) # read the grid from the HDF5 file see data_management_tools.py for further information

    n_areas = len(areas)

    # Generating a population
    if blackout:
        # assert that save != None
        population = save['pop']
        global_var.N_iter = save['N_iter']
    else:
        population = generate_pop(n_areas, global_var.N_pop, Range=global_var.init_range)

    # Optimization
    if optimisation_method == "MOPSO":
        velocities_init = [[0 for j in range(len(population[0]))] for i in range(len(population))]
        archive, memory = MOPSO(  population, 
                                  velocities_init, 
                                  global_var.N_iter, 
                                  lambda particle, penalty: mo_get_costs(particle, penalty, areas=areas), 
                                  weakly_dominate,
                                  archive_size=global_var.archive_size, 
                                  degrees_of_freedom=[global_var.Qlb, global_var.Qub], 
                                  p_turbulence=global_var.p_turbulence, penalty=global_var.penalty,
                                  output_memory_path=global_var.output_memory_path, live_saving=True,
                                  save=save)

    elif optimisation_method == "MOGA":
        memory = MOGA(  population, 
                        global_var.N_iter,
                        lambda particle, penalty: mo_get_costs(particle, penalty, areas=areas), 
                        weakly_dominate,
                        penalty=True,
                        crossover_method="unfair average",
                        degrees_of_freedom=[global_var.Qlb, global_var.Qub], 
                        mutation_method="non uniform", p_mutation=global_var.p_turbulence
                        )
    # Store outputs
    go_to_the_root()
    with open(global_var.output_pareto_opt_path, 'w', newline="") as counts_csv:
            writer = csv.writer(counts_csv)
            for i in range(global_var.N_iter):
                dico = memory[i]
                costsi = []
                for part_cost in dico["archive"]:
                    costsi.append(part_cost[1])
                writer.writerow(costsi)

    with open(global_var.output_costs_path, 'w', newline="") as counts_csv:
            writer = csv.writer(counts_csv)
            for i in range(global_var.N_iter):
                dico = memory[i]
                costsi = []
                for cost in dico["costs"]:
                    costsi.append(cost)
                writer.writerow(costsi)

    with open(global_var.clusters_file_path, 'w', newline="") as clus_csv:
        writer = csv.writer(clus_csv)
        for area in areas:
            writer.writerow([well.id for well in area.wells])

    if global_var.penalty:
        with open(global_var.output_penalties_path, 'w', newline="") as counts_csv:
                writer = csv.writer(counts_csv)
                for i in range(global_var.N_iter):
                    dico = memory[i]
                    costsi = []
                    for penalty in dico["penalties"]:
                        costsi.append(penalty)
                    writer.writerow(costsi)

    if display:
        display_map(areas, river)
        display_evolution(global_var.opti_folder)
