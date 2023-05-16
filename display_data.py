"""
Author Lilian Bosc 
Latest update: 13/12/2022

Functions to display the final results. Usefull to display the results and to understand and debug the code.

commented
"""

import global_var
import matplotlib.pyplot as plt
import csv
import os
import json
import time

def go_to_the_root():
    """
    The terminal goes to the root of the folder.
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("../")

def string2list(string):
    """
    string "[1, 2, 3]" -> list([1, 2, 3])
    """
    l = string.replace("[", "").replace("]", "").replace(" ", "").split(",")
    for i in range(len(l)):
        l[i] = float(l[i])
    return l

def weakly_dominate(x1, x2):
    """
    Parameters
    ----------
    x1, x2 are vectors [xi1, ... , xin]
    
    Output
    ------
    True if x1 weakly-dominates x2, False if not
    """
    if len(x1) != len(x2):
        raise Exception("x1 and x2 should have the same dimension.")
    n = len(x1)
    for k in range(n):
        if x1[k] < x2[k]:
            return False
    return True

def eps_weakly_dominate(x1, x2, eps):
    """
    Parameters
    ----------
    x1, x2 are vectors [xi1, ... , xin]
    eps float > 0
    
    Output
    ------
    True if x1 epsilon-weakly-dominates x2, False if not
    """
    if len(x1) != len(x2):
        raise Exception("x1 and x2 should have the same dimension.")
    n = len(x1)
    for k in range(n):
        if x1[k] < x2[k]*(1+eps):
            return False
    return True

def find_pareto_front(set_, greater):
    """
    Parameters
    ----------
    set: [[pi, [f1(pi), ..., fm(pi)]] for 0<i<N+1]
    or 
    set: [[pi, [f1(pi), ..., fm(pi)], penalty(i)] for 0<i<N+1]
    """
    pareto_front = []
    n = len(set_)
    m = len(set_[0][1])
    
    for i in range(n):
        flag = False
        j = 0
        while j<n and flag == False:
            if i != j:
                if greater(set_[j][1], set_[i][1]):
                    flag = True
            j += 1
        if flag == False:
            pareto_front.append(set_[i])
        
    return pareto_front

def display_pareto_front(optimisation, pause=0.2):
    """
    This function displays the Pareto front by recovering it from the folder data/optimisationi/model_output.

    Parameters
    ----------
    optimisation <string> : example "optimisation1" the name of the folder where the data are stored.
    """
    # Recovering the correct path
    go_to_the_root()
    opti_path = "data/" + optimisation
    output_path = ""
    for folder in os.listdir(opti_path):
        if "_output" in folder:
            output_path += opti_path + "/" + folder
    pareto_costs_file_path = ""
    for folder in os.listdir(output_path):
        if "pareto" in folder:
            pareto_costs_file_path += output_path + "/" + folder

    # Open and recover the data from the correct file
    with open(pareto_costs_file_path, 'r') as file:
        filecontent = csv.reader(file)
        rows = []
        for row in filecontent:
            rows.append(row)

    # stringto list
    for row in rows:
        for i in range(len(row)):
            row[i] = row[i].replace("[", "").replace("]", "").replace(",", "")
            row[i] = [float(row[i].split(" ")[0]), float(row[i].split(" ")[1])]

    # Plotting the results
    for i in range(len(rows)):
        row = rows[i]
        f1 = [el[0] for el in row]
        f2 = [el[1] for el in row]
        if (i+1)%3 == 0 or i==0 or i==len(rows)-1:
            plt.plot(f1, f2, "^", label=f"generation {i+1}")
        else:
            plt.plot(f1, f2, "^")
        plt.draw()
        plt.pause(pause)
        plt.legend()
        # plt.clf()
    plt.plot(f1, f2, "^", label=f"generation {i+1}")

    n_pop = len(rows[0])
    n_iter = len(rows)
    
    plt.title(f"Pareto front: N_iter={n_iter}; N_pop={n_pop}")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.legend(loc="lower left")
    plt.show()

def display_poplation(optimisation, pause=0.1):
    # Displays the evolution of the population for all the generations.
    go_to_the_root()
    opti_path = "data/" + optimisation
    output_path = ""
    for folder in os.listdir(opti_path):
        if "_output" in folder:
            output_path += opti_path + "/" + folder
    costs_path = ""
    for folder in os.listdir(output_path):
        if "costs" in folder:
            costs_path += output_path + "/" + folder

    with open(costs_path, 'r') as file:
        filecontent = csv.reader(file)
        rows = []
        for row in filecontent:
            rows.append(row)

    for row in rows:
        for i in range(len(row)):
            row[i] = row[i].replace("[", "").replace("]", "").replace(",", "")
            row[i] = [float(row[i].split(" ")[0]), float(row[i].split(" ")[1])]
    max_x = max([max([row[i][0] for i in range(len(row))]) for row in rows])
    min_x = min([min([row[i][0] for i in range(len(row))]) for row in rows])
    max_y = max([max([row[i][1] for i in range(len(row))]) for row in rows])
    min_y = min([min([row[i][1] for i in range(len(row))]) for row in rows])
    gap_x = (max_x-min_x)/10
    gap_y = (max_y-min_y)/10
    for i in range(len(rows)):
        plt.clf()
        row = rows[i]
        f1 = [el[0] for el in row]
        f2 = [el[1] for el in row]
        
        plt.plot(f1, f2, "o", label=f"generation {i+1}")
        plt.ylim(min_y-gap_y, max_y+gap_y)
        plt.xlim(min_x-gap_x, max_x+gap_x)
        plt.draw()
        plt.pause(pause)
        plt.legend()
    
    n_pop = len(rows[0])
    n_iter = len(rows)
    
    plt.title(f"Swarm evo: N_iter={n_iter}; N_pop={n_pop}")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.show()

def display_penalties(optimisation, pause=0.1):
    # displays the evolution of the penalties for all the generations.
    go_to_the_root()
    opti_path = "data/" + optimisation
    output_path = ""
    for folder in os.listdir(opti_path):
        if "_output" in folder:
            output_path += opti_path + "/" + folder
    penalties_file_path = ""
    for folder in os.listdir(output_path):
        if "penalties" in folder:
            penalties_file_path += output_path + "/" + folder

    with open(penalties_file_path, 'r') as file:
        filecontent = csv.reader(file)
        rows = []
        for row in filecontent:
            rows.append(row)
    for row in rows:
        for i in range(len(row)):
            row[i] = float(row[i])
        print(row)
    max_x = len(rows)
    min_x = 0
    max_y = max([max([row[i] for i in range(len(row))]) for row in rows])
    min_y = min([min([row[i] for i in range(len(row))]) for row in rows])
    gap_x = (max_x-min_x)/10
    gap_y = (max_y-min_y)/10

    for i in range(len(rows)):
        row = rows[i]
        X = [i for _ in range(len(row))]
        plt.plot(X, row, "_")
        plt.ylim(min_y-gap_y, max_y+gap_y)
        plt.xlim(min_x-gap_x, max_x+gap_x)
        plt.draw()
        plt.pause(pause)
    
    n_pop = len(rows[0])
    n_iter = len(rows)

    plt.title(f"Penalties evo: N_iter={n_iter}; N_pop={n_pop}")
    plt.xlabel("Generation")
    plt.ylabel("Penalty")
    plt.show()


def display_evolution(optimisation, pause=0.1):
    # Displays the evolution of all the parameters across generations.
    go_to_the_root()
    opti_path = "data/" + optimisation

    output_path = ""
    for folder in os.listdir(opti_path):
        if "_output" in folder:
            output_path += opti_path + "/" + folder

    penalties_file_path = ""
    costs_path = ""
    pareto_costs_file_path = ""
    for file in os.listdir(output_path):
        if "penalties" in file:
            penalties_file_path += output_path + "/" + file
        if "costs" in file:
            costs_path += output_path + "/" + file
        if "pareto" in file:
            pareto_costs_file_path += output_path + "/" + file

    with open(penalties_file_path, 'r') as file:
        filecontent = csv.reader(file)
        penalties_rows = []
        for row in filecontent:
            penalties_rows.append(row)

    with open(costs_path, 'r') as file:
        filecontent = csv.reader(file)
        costs_rows = []
        for row in filecontent:
            costs_rows.append(row)

    with open(pareto_costs_file_path, 'r') as file:
        filecontent = csv.reader(file)
        pareto_rows = []
        for row in filecontent:
            pareto_rows.append(row)

    # Penalties data treatment
    for row in penalties_rows:
        for i in range(len(row)):
            row[i] = float(row[i])
    p_max_x = len(penalties_rows)
    p_min_x = 0
    p_max_y = max([max([row[i] for i in range(len(row))]) for row in penalties_rows])
    p_min_y = min([min([row[i] for i in range(len(row))]) for row in penalties_rows])
    p_gap_x = (p_max_x-p_min_x)/10
    p_gap_y = (p_max_y-p_min_y)/10

    # Costs data treatment
    for row in costs_rows:
        for i in range(len(row)):
            row[i] = row[i].replace("[", "").replace("]", "").replace(",", "")
            row[i] = [float(row[i].split(" ")[0]), float(row[i].split(" ")[1])]
    max_x = max([max([row[i][0] for i in range(len(row))]) for row in costs_rows])
    min_x = min([min([row[i][0] for i in range(len(row))]) for row in costs_rows])
    max_y = max([max([row[i][1] for i in range(len(row))]) for row in costs_rows])
    min_y = min([min([row[i][1] for i in range(len(row))]) for row in costs_rows])
    gap_x = (max_x-min_x)/10
    gap_y = (max_y-min_y)/10

    # Pareto data treatment
    for row in pareto_rows:
        for i in range(len(row)):
            row[i] = row[i].replace("[", "").replace("]", "").replace(",", "")
            row[i] = [float(row[i].split(" ")[0]), float(row[i].split(" ")[1])]


    figure, axis = plt.subplots(2, 2, figsize=(9,5))

    for i in range(len(penalties_rows)):
        # Swarm movement
        row = costs_rows[i]
        f1_ = [el[0] for el in row]
        f2_ = [el[1] for el in row]
        
        axis[0,0].clear()
        axis[0,0].plot(f1_, f2_, "b.")
        axis[0,0].set_ylim([min_y-gap_y, max_y+gap_y])
        axis[0,0].set_xlim([min_x-gap_x, max_x+gap_x])
        
        # Swarm print
        c_row = costs_rows[i]
        f1 = [el[0] for el in c_row]
        f2 = [el[1] for el in c_row]
        
        axis[0,1].plot(f1, f2, "b,")
        axis[0,1].set_ylim([min_y-gap_y, max_y+gap_y])
        axis[0,1].set_xlim([min_x-gap_x, max_x+gap_x])

        
        # Pareto front

        pa_row = pareto_rows[i]
        f1 = [el[0] for el in pa_row]
        f2 = [el[1] for el in pa_row]
        
        axis[1,0].plot(f1, f2, "^")
        
        # Penalties
        p_row = penalties_rows[i]
        X = [i for _ in range(len(p_row))]
        axis[1, 1].plot(X, p_row, "r_")
        axis[1, 1].set_ylim([p_min_y-p_gap_y, p_max_y+p_gap_y])
        axis[1, 1].set_xlim([p_min_x-p_gap_x, p_max_x+p_gap_x])
        
        
        # Combine all the operations and display
        plt.draw()
        plt.pause(pause)
    plt.show()

def display_pf_old(list_folders_algos, eps=0):
    # Displays the evolution of the pareto fronts generations after generations
    for name_folder, algo in list_folders_algos:
        print('parsing '+name_folder+'...')
        go_to_the_root()
        opti_path = "data/" + name_folder

        output_path = ""
        for folder in os.listdir(opti_path):
            if "_output" in folder:
                output_path += opti_path + "/" + folder
        for file in os.listdir(output_path):
            if "memory" in file:
                memory_file_path = output_path + "/" + file
        with open(memory_file_path, 'r') as file:
            filecontent = file.read()

        list_gen = filecontent.split("\n____________________________________________________________________________________________________________________________\n")[:-1]   
        all_pfs = {}
        for gen in range(len(list_gen)):
            el = list_gen[gen]
            sep_pf = el.split("pareto front:\n")[1].split("\n")
            particles = [string2list(sep_pf[3*i]) for i in range(len(sep_pf)//3)]
            costs = [string2list(sep_pf[3*i+1].replace("costs: ", "")) for i in range(len(sep_pf)//3)]
            penalties = [float(sep_pf[3*i+2].replace("penalty: ", "")) for i in range(len(sep_pf)//3)]

            all_pfs[gen] = {
                "particles" : particles,
                "costs" : costs,
                "penalties" : penalties,
            }
        all_popts = []
        for gen in all_pfs.keys():
            for i in range(len(all_pfs[gen]["particles"])):
                all_popts.append([all_pfs[gen]["particles"][i], all_pfs[gen]["costs"][i]])
        
        popts_oat = find_pareto_front(all_popts, lambda x, y: eps_weakly_dominate(x, y, eps))
        
        with open(output_path+"/pf-variables.csv", 'w', newline="") as file:
            writer = csv.writer(file)
            for part in popts_oat:
                writer.writerow(part[0])
        with open(output_path+"/pf-costs.csv", 'w', newline="") as file:
            writer = csv.writer(file)
            for part in popts_oat:
                writer.writerow(part[1])

        X = [el[1][0] for el in popts_oat]
        Y = [el[1][1] for el in popts_oat]
        
        plt.plot(X, Y, ".", label=algo)
    plt.legend()
    plt.show()    

def display_pf(list_folders_algos):
    # displays pareto front
    for name_folder, algo in list_folders_algos:
        print('parsing '+name_folder+'...')
        go_to_the_root()
        opti_path = "data/" + name_folder

        output_path = ""
        for folder in os.listdir(opti_path):
            if "_output" in folder:
                output_path += opti_path + "/" + folder
        for file in os.listdir(output_path):
            if "memory" in file:
                memory_file_path = output_path + "/" + file
        with open(memory_file_path, 'r') as file:
            filecontent = file.read()

        list_gen = filecontent.split("\n____________________________________________________________________________________________________________________________\n")[:-1]   
        all_pfs = {}

        el = list_gen[-1]
        sep_pf = el.split("pareto front:\n")[1].split("\n")
        particles = [string2list(sep_pf[3*i]) for i in range(len(sep_pf)//3)]
        costs = [string2list(sep_pf[3*i+1].replace("costs: ", "")) for i in range(len(sep_pf)//3)]
        penalties = [float(sep_pf[3*i+2].replace("penalty: ", "")) for i in range(len(sep_pf)//3)]

        pf = {
                "particles" : particles,
                "costs" : costs,
                "penalties" : penalties,
            }

        with open(output_path+"/pf-variables.csv", 'w', newline="") as file:
            writer = csv.writer(file)
            for part in pf["particles"]:
                writer.writerow(part)
        with open(output_path+"/pf-costs.csv", 'w', newline="") as file:
            writer = csv.writer(file)
            for part in pf["costs"]:
                writer.writerow(part)

        X = [el[0] for el in pf["costs"]]
        Y = [el[1] for el in pf["costs"]]
        
        plt.plot(X, Y, "^", label=algo)
    plt.legend()
    plt.show()

def display_avancement(list_folders):
    go_to_the_root()
    flag = True
    while flag:
        tobe_printed = []
        for folder in list_folders:
            opti_path = "data/" + folder
            for file in os.listdir(opti_path):
                if '_input_variables.json' in file:
                    input_file = opti_path + "/" + file
                if 'output' in file:
                    output_folder = opti_path + "/" + file
            for file in os.listdir(output_folder):
                if "memory" in file:
                    memory_path = output_folder + '/' + file

            with open(input_file) as f:
                variables = json.load(f)
            with open(memory_path, 'r') as f:
                mem = f.read()
            
            N_iter = variables["N_iter"]
            t = len(mem.split('t = '))-1
            tobe_printed.append((folder, t, N_iter))
        flag = False
        for folder, t, N_iter in tobe_printed:
            if t != N_iter:
                flag = True
            print(f'{folder}: {t}/{N_iter}')
        time.sleep(2)

