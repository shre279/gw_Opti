"""
Author: Lilian Bosc
Latest update: 06/06/2022

Tool for Multi Objective optimisation (particle swarm optimisation)
With a turbulence.

commented
"""

# from setup import *
from modules import *
from data_management_tools import *


# Side functions

def dominate_max(x1, x2):
    """
    Parameters
    ----------
    x1, x2 are vectors [xi1, ... , xin]
    
    Output
    ------
    True if x1 dominates x2, False if not
    """
    if len(x1) != len(x2):
        raise Exception("x1 and x2 should have the same dimension.")
    n = len(x1)
    for k in range(n):
        if x1[k] <= x2[k]:
            return False
    return True

    
def weakly_dominate_max(x1, x2):
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
    if x1 == x2:
        return True
    n = len(x1)
    for k in range(n):
        if x1[k] < x2[k]:
            return False
    return True

def weakly_dominate_min(x1, x2):
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
    if x1 == x2:
        return True
    n = len(x1)
    for k in range(n):
        if x1[k] > x2[k]:
            return False
    return True

def eps_weakly_dominate_max(x1, x2, eps):
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
    if x1 == x2:
        return False
    n = len(x1)
    for k in range(n):
        if x1[k] < x2[k](1+eps):
            return False
    return True

def evaluate(pop, mo_fitness_function, penalty):
    Ct = []
    penalties = []
    for i in range(len(pop)):
        if penalty:
            costs_i, penalty_i = mo_fitness_function(pop[i], penalty)
            Ct.append(costs_i)
            penalties.append(penalty_i)
        else:
            Ct.append(mo_fitness_function(pop[i]))
    if penalty:
        return Ct, penalties
    return Ct

def find_pareto_front(set_, comp_tool):
    """
    Find the non-dominated values
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
        if i == 0:
            j = 1
        else:
            j = 0
        while j<n and not comp_tool(set_[j][1], set_[i][1]):
            if j == i-1:
                j += 2
            else:
                j += 1
        if j == n:
            pareto_front.append(set_[i])
    return pareto_front

def crowding_distance(particle, pf):
    """
    return the crowding distance score of the particle
    Parameter
    ---------
    index: int
    pf: array [[p1, cost, penalty], ...]

    Output
    ------
    crowding_distance
    """
    new_pf = copy.deepcopy(pf)
    new_pf.remove(particle)
    cost = particle[1]
    axis_distance = []
    for axis in range(len(cost)):
        mini = abs(new_pf[0][1][axis] - cost[axis])
        for i in range(len(new_pf)):
            dist = abs(pf[i][1][axis] - cost[axis])
            if dist < mini:
                mini = dist
        axis_distance.append(mini)
    return sum(axis_distance)
    
def m(left, right, pf):
    if not len(left) or not len(right):
        return left or right
 
    result = []
    i, j = 0, 0
    while (len(result) < len(left) + len(right)):
        if crowding_distance(left[i], pf) < crowding_distance(right[j], pf):
            result.append(left[i])
            i+= 1
        else:
            result.append(right[j])
            j+= 1
        if i == len(left) or j == len(right):
            result.extend(left[i:] or right[j:])
            break
    return result
 
def sort_pf_cd(list, pf):
    if len(list) < 2:
        return list
 
    middle = int(len(list)/2)
    left = sort_pf_cd(list[:middle], pf)
    right = sort_pf_cd(list[middle:], pf)
 
    return m(left, right, pf)

def clean_pf_cd(pf, max_particles):
    """
    return an homogeneous pareto front if the number of particle exceed max_part using crowding distance (cd)
    Parameter
    ---------
    max_part: int
    pf: array [[p1, cost, penalty], ...]

    Output
    ------
    clean_pf
    """
    new_pf = sort_pf_cd(pf, pf)
    return new_pf[:max_particles]

def join(set1, set2):
    new_set = copy.deepcopy(set1)
    for el in set2:
        new_set.append(el)
    return new_set

def complete(pf_, set_, lb):
    forbidden = []
    for el in pf_:
        forbidden.append(set_.index(el))
    choices = []
    for i in range(len(set_)):
        if i not in forbidden:
            choices.append(i)
    indexes = random.sample(choices, min(len(choices), lb-len(pf_)))
    for index in indexes:
        pf_.append(set_[index])

def fulfill(set_, new_set_, lb, comp_tool):
    SET_ = join(set_, new_set_)
    pf_ = find_pareto_front(SET_, comp_tool)
    if len(pf_) < lb:
        complete(pf_, SET_, lb)
    return pf_
        
def update_ARC(archive, pop, Ct, comp_tool, lb, ub):
    # Find the pf in the population
    set_ = [[pop[i], Ct[i]] for i in range(len(pop))]
    pf = find_pareto_front(set_, comp_tool)

    # Add the pf to the archive
    new_archive = fulfill(archive, pf, lb, comp_tool)

    # Use crowding distance reduction
    if len(new_archive) > ub:
        new_archive = clean_pf_cd(new_archive, ub)
    print(len(new_archive))
    return new_archive

def update_ARC_with_penalty(archive, pop, Ct, penalties, comp_tool, lb, ub):
    # Find the pf in the population
    set_ = [[pop[i], Ct[i], penalties[i]] for i in range(len(pop))]
    pf = find_pareto_front(set_, comp_tool)

    # Add the pf to the archive
    new_archive = fulfill(archive, pf, lb, comp_tool)

    # Use crowding distance reduction
    if len(new_archive) > ub:
        new_archive = clean_pf_cd(new_archive, ub)
    print(len(new_archive))
    return new_archive

def sigma_method(vect, set_):
    """
    In 2D only, to improve...
    sigma method to find the Global best of a particle
    Parameters
    ----------
    vect: <array> [f1, ..., fm]
    set_: <array> [[pi, [f1(pi), ..., fm(pi)]] for 0<i<N+1]
    or 
    set_: <array> [[pi, [f1(pi), ..., fm(pi)], penalty(i)] for 0<i<N+1]
    """
    c0 = set_[0][1]
    if vect[0]**2+vect[1]**2 == 0:
        if c0[0]**2+c0[1]**2 == 0:
            mindist = 0
        else:
            mindist =  abs(c0[0]**2-c0[1]**2)/(c0[0]**2+c0[1]**2)
    elif c0[0]**2+c0[1]**2 == 0:
        mindist = abs(vect[0]**2-vect[1]**2)/(vect[0]**2+vect[1]**2)
    else:
        mindist = abs((vect[0]**2-vect[1]**2)/(vect[0]**2+vect[1]**2) - (c0[0]**2-c0[1]**2)/(c0[0]**2+c0[1]**2))
    GB = set_[0][0]
    for i in range(1,len(set_)):
        ci = set_[i][1]
        if vect[0]**2+vect[1]**2 == 0:
            if ci[0]**2+ci[1]**2 == 0:
                dist = 0
            else:
                dist =  abs(ci[0]**2-ci[1]**2)/(ci[0]**2+ci[1]**2)
        elif ci[0]**2+ci[1]**2 == 0:
            dist = abs(vect[0]**2-vect[1]**2)/(vect[0]**2+vect[1]**2)
        else:
            dist = abs((vect[0]**2-vect[1]**2)/(vect[0]**2+vect[1]**2) - (ci[0]**2-ci[1]**2)/(ci[0]**2+ci[1]**2))
        
        if dist < mindist:
            mindist = dist
            GB = set_[i][0]
    return GB

def generate(pop, Ct, archive, vel, PB, w, c1, c2, mode, degrees_of_freedom, p_turbulence):
    n = len(pop)
    if mode not in ["freewill", "swarm"]:
        raise Exception("Unrecognized mode. Should be 'freewill' or 'swarm'.")
    if mode == "swarm":
        r1 = random.random()
        r2 = random.random()
        
    if degrees_of_freedom == []:
        lb = -float("inf")
        ub = float("inf")
    elif type(degrees_of_freedom) == list and len(degrees_of_freedom) == 2:
        lb, ub = degrees_of_freedom[0], degrees_of_freedom[1]
    else:
        raise Exception("degrees of freedom must be [] or [lb, ub]")

    for i in range(n):
        if mode == "freewill":
            r1 = random.random()
            r2 = random.random()
        GBi = sigma_method(Ct[i], archive)
        for j in range(len(pop[i])):
            vel[i][j] = w*vel[i][j] + r1*c1*(PB[i][j] - pop[i][j]) + r2*c2*(GBi[j] - pop[i][j])
            pop[i][j] += vel[i][j]
            if pop[i][j] > ub:
                pop[i][j] = ub
            if pop[i][j] < lb:
                pop[i][j] = lb

            if random.random() < p_turbulence:
                done = False
                while not done:
                    neg = random.random() < 0.5
                    if neg:
                        turbulence = -random.random()
                    else:
                        turbulence = random.random()
                    pij = (1+turbulence)*pop[i][j]
                    if lb <= pij <= ub:
                        pop[i][j] = pij
                        done = True
                
        PB[i] = GBi
    return PB, pop


def MOPSO( pop_init, 
            vel_init, 
            N_iter, 
            mo_fitness_function, 
            comp_tool,
            w=0.7, c1=1, c2=2, penalty=False, degrees_of_freedom= [], p_turbulence=0, 
            mode="swarm", archive_size="auto", output_memory_path=global_var.output_memory_path,
            live_saving=False, save=None):
    """
    This function run a MOPSO model

    Parameters
    ----------
    pop_init: <array> of any kind that is compatible with the fitness function
    vel_init: <array> of any kinf that is compatible with population
    N_iter: <int> number of iterations
    mo_fitness_function: <function> that should alway be to maximise (by the domination_comparator chosen)
    comp_tool: <function> that is able to compare two vectors
    N_obj: <int> number of objectives to optimize
    w: <float> inertia parameter
    c1: <float> constant linked to the personal best
    c2: <float> constant linked to the global best
    save_penalty: <boolean> True if the mo_fitness_function use a penalty to judge the particles
    degree_of_freedom: <list> [] or [lower_bound, upper_bound] for the decision parameters
    mode: <str> different mode of raising the random values
    archive_size: "auto" or a range [lb, ub]

    Outputs
    -------
    pareto_front : <array> set of pareto optimal particles
    archive : <array> the archive of all the pareto optimal particles
    costs_archive: <array> the archive of the objectives functions
    if there is a penalty:
    penalties_archive: <array> the archive of all the penalties
    memory: dict {
                    [t] : {
                            "pop",
                            "archive",
                            "costs",
                            "penalties"
                            } 
                 }

    Options
    -------

    Author: Lilian Bosc, 01/06/2022
    """
    pop = pop_init
    vel = vel_init
    n_pop = len(pop)
    PB = [pop[random.randrange(n_pop)] for _ in range(n_pop)]
    memory = {}
    penalties = []

    if save == None:
        archive = []
        m = 0
    else:
        memory_txt = save['memory']
        m = len(memory_txt)
        archive = save['archive']
        
    if penalty:
        memory[0] = {
                    "pop": copy.deepcopy(pop),
                    "archive": copy.deepcopy(archive),
                     "penalties": copy.deepcopy(penalties)
                    }
    else:
        memory[0] = {
                    "pop": copy.deepcopy(pop),
                    "archive": copy.deepcopy(archive)
                    }
        
    if archive_size == "auto":
        ub = 100
    elif type(archive_size) == int:
        ub = archive_size
    else:
        raise Exception("archive_size must be 'auto' or int")
    lb = max(1, int(n_pop*0.2))
    #######################################################################################
    for t in range(N_iter):
        print(f"generation: {t}/{N_iter}")
        if penalty:
            Ct, penalties = evaluate(pop, mo_fitness_function, penalty)
            archive = update_ARC_with_penalty(archive, pop, Ct, penalties, comp_tool, lb, ub)
        else:
            Ct = evaluate(pop, mo_fitness_function, penalty)
            archive = update_ARC(archive, pop, Ct, comp_tool, lb, ub)
        PB, pop = generate(pop, Ct, archive, vel, PB, w, c1, c2, mode, degrees_of_freedom, p_turbulence)

    ########################################################################################    
        memory[t+1] = {
                 "pop": copy.deepcopy(pop),
                 "archive": [],
                 "costs": []
                }
        if penalty:
            memory[t]["penalties"] = copy.deepcopy(penalties)
        memory[t]["archive"] = copy.deepcopy(archive)
        memory[t]["costs"] = copy.deepcopy(Ct)

        with open(output_memory_path, 'a', newline="") as file:
            file.write(f"t = {m + t}\n")
            file.write("population:\n")
            for part in memory[t]["pop"]:
                file.write(str(part)+"\n")
            file.write("\npareto front:\n")
            for arch in memory[t]["archive"]:
                file.write(str(arch[0])+"\n")
                file.write("costs: " + str(arch[1])+"\n")
                if penalty:
                    file.write("penalty: " + str(arch[2])+"\n")
            file.write("____________________________________________________________________________________________________________________________\n")
        
        if live_saving:
            # in case of blackout
            with open(global_var.output_pareto_opt_path, 'a', newline="") as counts_csv:
                writer = csv.writer(counts_csv)
                dico = memory[t]
                costsi = []
                for part_cost in dico["archive"]:
                    costsi.append(part_cost[1])
                writer.writerow(costsi)

            with open(global_var.output_costs_path, 'a', newline="") as counts_csv:
                writer = csv.writer(counts_csv)
                dico = memory[t]
                costsi = []
                for cost in dico["costs"]:
                    costsi.append(cost)
                writer.writerow(costsi)
    return archive, memory

