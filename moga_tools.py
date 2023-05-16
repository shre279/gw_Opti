"""
Author Lilian Bosc
Date 16/06/2022

Tools for multi-objective optimisation (genetic algorithm)

commented
"""
from copy import deepcopy
from modules import *
from data_management_tools import generate_pop

def dominate(x1, x2):
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

def eps_dominate(x1, x2, eps):
    """
    Parameters
    ----------
    x1, x2 are vectors [xi1, ... , xin]
    eps float > 0
    
    Output
    ------
    True if x1 epsilon-dominates x2, False if not
    """
    if len(x1) != len(x2):
        raise Exception("x1 and x2 should have the same dimension.")
    n = len(x1)
    for k in range(n):
        if x1[k] <= x2[k]/(1+eps):
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

def find_all_pfs(set_, greater):
    pfs = []
    new_set = deepcopy(set_)
    while len(new_set) > 0:
        pf = find_pareto_front(new_set, greater)
        if pf == []:
            pfs.append(new_set)
            new_set = []
        else:
            pfs.append(pf)
            for el in pf:
                new_set.remove(el)
    return pfs

def naive_crossover(p1, p2):
    n = len(p1)
    l = random.randint(1, n-1)
    child1 = [p2[i] for i in range(n)]
    child2 = [p1[i] for i in range(n)]
    for i in range(l):
        child1[i] = p1[i]
        child2[i] = p2[i]
    return child1, child2

def flip_coin_crossover(p1, p2):
    n = len(p1)
    child1 = []
    child2 = []
    for i in range(n):
        xi1 = random.choice([p1[i], p2[i]])
        xi2 = [p1[i], p2[i]].remove(xi1)[0]
        child1.append(xi1)
        child2.append(xi2)
    return child1, child2

def insertion_crossover(p1, p2):
    n = len(p1)
    lb = random.randint(0, n-1)
    ub = random.randint(0, n-1)
    while lb == ub:
        lb = random.randint(0, n-1)
    if lb > ub:
        lb, ub = ub, lb
    child1 = [p1[i] for i in range(n)]
    child2 = [p2[i] for i in range(n)]
    for i in range(lb, ub):
        child1[i] = p2[i]
        child2[i] = p1[i]
    return child1, child2

def blend_crossover(p1, p2, alpha, degrees_of_freedom):
    n = len(p1)
    child = []
    for i in range(n):
        # create interval
        xi1 = p1[i]
        xi2 = p2[i]
        lb = min(xi1, xi2) - alpha*abs(xi1-xi2)
        ub = max(xi1, xi2) + alpha*abs(xi1-xi2)
        new_gen = (ub-lb)*random.random()+lb
        new_gen = max(degrees_of_freedom[0], min(new_gen, degrees_of_freedom[1]))
        child.append(new_gen) # random number in [lb, ub]

    return [child]

def unfair_avg_crossover(p1, p2, alpha, degrees_of_freedom):
    n = len(p1)
    child1 = []
    child2 = []
    j = random.randint(1, n-1)
    for i in range(n):
        if i <= j:
            new_gen1 = (1+alpha)*p1[i] - alpha*p2[i]
            new_gen2 = (1-alpha)*p1[i] + alpha*p2[i]
            new_gen1 = max(degrees_of_freedom[0], min(new_gen1, degrees_of_freedom[1]))
            new_gen2 = max(degrees_of_freedom[0], min(new_gen2, degrees_of_freedom[1]))
            child1.append(new_gen1)
            child2.append(new_gen2)
        else:
            new_gen1 = -alpha*p1[i] + (1+alpha)*p2[i]
            new_gen2 = alpha*p1[i] + (1-alpha)*p2[i]
            new_gen1 = max(degrees_of_freedom[0], min(new_gen1, degrees_of_freedom[1]))
            new_gen2 = max(degrees_of_freedom[0], min(new_gen2, degrees_of_freedom[1]))
            child1.append(new_gen1)
            child2.append(new_gen2)
    return child1, child2

def naive_mutation(p):
    n = len(p)
    i1 = random.randint(0, n-1)
    i2 = random.randint(0, n-1)
    while i1 == i2:
        i2 = random.randint(0, n-1)
    p[i1], p[i2] = p[i2], p[i1]

def non_uniform_mutation(p, boundaries, gen, max_gen, b, degrees_of_freedom):
    n = len(p)
    i_m = random.randint(0, n-1)
    lb, ub = boundaries[i_m]
    mutated = p[i_m] + random.choice([-1, 1])*(ub-lb)*(1 - random.random()**(1 - (gen/max_gen)**b))
    p[i_m] = max(degrees_of_freedom[0], min(degrees_of_freedom[1], mutated))

def normally_distribute_mutation(p, sigma, degrees_of_freedom):
    n = len(p)
    i_m = random.randint(0, n-1)
    sigma_i = sigma[i_m]
    mutated = p[i_m] + (random.random()-0.5)*sigma_i
    p[i_m] = max(degrees_of_freedom[0], min(degrees_of_freedom[1], mutated))

def reproduce(p1, p2, p_mut, crossover_method="naive", crossover_parameters={}, mutation_method="naive", mutation_parameters={}):
    p1, p2 = p1[0], p2[0]
    n = len(p1)
    if "search space boundaries" in mutation_parameters.keys():
        degrees_of_freedom = mutation_parameters["search space boundaries"]
    elif "search space boundaries" in crossover_parameters.keys():
        degrees_of_freedom = crossover_parameters["search space boundaries"]
    else:
        degrees_of_freedom = [float('-inf'), float('inf')]
    # gens distribution
    descendance = []
    if crossover_method == "naive":
        descendance = naive_crossover(p1, p2)
    elif crossover_method == "flip coin":
        descendance = flip_coin_crossover(p1, p2)
    elif crossover_method == "insertion":
        descendance = insertion_crossover(p1, p2)
    elif crossover_method == "blender":
        if "alpha" not in crossover_parameters.keys():
            alpha = 0.5
        else:
            alpha =crossover_parameters["alpha"]
        descendance = blend_crossover(p1, p2, alpha, degrees_of_freedom)
    elif crossover_method == "unfair average":
        if "alpha" not in crossover_parameters.keys():
            alpha = 0.5
        else:
            alpha = crossover_parameters["alpha"]
        descendance = unfair_avg_crossover(p1, p2, alpha, degrees_of_freedom)

    # mutation
    for child in descendance:
        if random.random() < p_mut:
            if mutation_method == "naive":
                naive_mutation(child, degrees_of_freedom)
            elif mutation_method == "non uniform":
                if "b" not in mutation_parameters.keys():
                    b = 1
                else:
                    b = mutation_parameters["b"]
                if "search space boundaries" not in mutation_parameters.keys():
                    # We will take a pourcentage of each decision variable of child
                    boundaries = [[0, child[i]] for i in range(n)]
                else:
                    boundaries = [mutation_parameters["search space boundaries"] for _ in range(n)]
                    # to improve if not the same boundaries on all decision variables
                if "gen" not in mutation_parameters.keys():
                    gen = 0
                else:
                    gen = mutation_parameters["gen"]
                if "max gen" not in mutation_parameters.keys():
                    max_gen = 1
                else:
                    max_gen = mutation_parameters["max gen"]
                non_uniform_mutation(child, boundaries, gen, max_gen, b, degrees_of_freedom)
            elif mutation_method == "normal":
                if "sigma" in mutation_parameters.keys():
                    sigma = mutation_parameters["sigma"]
                else:
                    sigma = [[child[i]*0.5] for i in range(n)]
                normally_distribute_mutation(child, sigma, degrees_of_freedom)
    return descendance

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
 
def msort(list, pf):
    if len(list) < 2:
        return list
 
    middle = int(len(list)/2)
    left = msort(list[:middle], pf)
    right = msort(list[middle:], pf)
 
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
    new_pf = msort(pf, pf)
    return new_pf[:max_particles]
    
def pick_a_pf(K):
    if K == 0:
        return 0
    flag = False
    while not flag:
        pick = []
        p = 0.6
        for k in range(K):
            if random.random() < p*1.001**k:
                pick.append(k)
                break
        if pick != []:
            flag = True
    return pick[0]

def pick_and_remove_elite(pfs, n_pf_empty=0):
    K = len(pfs)
    k = pick_a_pf(K-n_pf_empty)+(n_pf_empty-1)
    if len(pfs[k]) > 0:
        elite = random.choice(pfs[k])
        pfs[k].remove(elite)
        return elite
    else:
        return pick_and_remove_elite(pfs, n_pf_empty=n_pf_empty+1)
    
def select(pop, costs, greater, selection_rate):
    # construct set_
    set_ = []
    for i in range(len(pop)):
        set_.append([pop[i], costs[i]])
    # print(set_)
    pfs = find_all_pfs(set_, greater)
    K = len(pfs)
    max_pf_particles = int(len(pop)*0.5)
    for i in range(len(pfs)):
        if len(pfs[i]) > max_pf_particles:
            pfs[i] = clean_pf_cd(pfs[i], max_pf_particles)
            
    new_pop = []
    # pick pareto fronts
    while len(new_pop)<len(pop)*selection_rate:
        new_elite = pick_and_remove_elite(pfs)
        new_pop.append(new_elite)
    return new_pop

def select_with_penalties(pop, costs, penalties, greater, selection_rate):
    # construct set_
    set_ = []
    for i in range(len(pop)):
        set_.append([pop[i], costs[i], penalties[i]])
    # print(set_)
    pfs = find_all_pfs(set_, greater)
    K = len(pfs)
    max_pf_particles = int(len(pop)*0.5)
    for i in range(len(pfs)):
        if len(pfs[i]) > max_pf_particles:
            pfs[i] = clean_pf_cd(pfs[i], max_pf_particles)
            
    new_pop = []
    # pick pareto fronts
    while len(new_pop)<len(pop)*selection_rate:
        new_elite = pick_and_remove_elite(pfs)
        new_pop.append(new_elite)
    return new_pop
 
def restock(reproducers, n_pop, p_mut, crossover_method, crossover_parameters, mutation_method, mutation_parameters):
    new_pop = []
    random.shuffle(reproducers)
    i = 0
    while len(new_pop) < n_pop:
        p1 = reproducers[i%len(reproducers)]
        p2 = reproducers[(i+1)%len(reproducers)]
        descendance = reproduce(p1, p2, 
                                p_mut, 
                                crossover_method,
                                crossover_parameters,
                                mutation_method,
                                mutation_parameters
                                )
        for child in descendance:
            new_pop.append(child)
        if i == len(reproducers):
            random.shuffle(reproducers)
        i += 1
    return new_pop


def MOGA(  pop_init, 
                N_iter, 
                mo_fitness_function, 
                domination_greater,
                crossover_method="naive",
                mutation_method="naive",
                crossover_parameters={},
                mutation_parameters={},
                penalty=False, degrees_of_freedom= [], selection_rate=0.5, 
                p_mutation=0.1, output_memory_path=global_var.output_memory_path
                ):
    if degrees_of_freedom != []:
        mutation_parameters["search space boundaries"] = degrees_of_freedom # a problem can come from here
        mutation_parameters["sigma"] = [abs(degrees_of_freedom[1]-degrees_of_freedom[0])*0.5 for _ in range(len(pop_init[0]))]
    mutation_parameters["max gen"] = N_iter
    # STP 1: initialisation
    memory = {}
    N_pop = len(pop_init)
    pop = pop_init
    penalties = []
    # file = open(global_var.output_memory_path, "w")
    # file.close()
    memory[0] = {"pop": copy.deepcopy(pop)}

    for gen in range(N_iter):
        mutation_parameters["gen"] = gen
        if penalty:
            print("evaluate...")
            Ct, penalties = evaluate(pop, mo_fitness_function, penalty)
            print("select...")
            reproducers = select_with_penalties(pop, Ct, penalties, domination_greater, selection_rate)
            print("restock...")
            pop = restock(  reproducers, 
                            N_pop,
                            p_mutation, 
                            crossover_method,
                            crossover_parameters,
                            mutation_method,
                            mutation_parameters)

        else:
            print("evaluate...")
            Ct = evaluate(pop, mo_fitness_function, penalty)
            print("select...")
            reproducers = select(pop, Ct, domination_greater, selection_rate)
            print("restock...")
            pop = restock(  reproducers, 
                            N_pop,
                            p_mutation, 
                            crossover_method,
                            crossover_parameters,
                            mutation_method,
                            mutation_parameters)
        
        # Archive and show
        memory[gen+1] = {
                 "pop": copy.deepcopy(pop),
                 "archive": [],
                 "costs": []
                }
        if penalty:
            memory[gen]["penalties"] = copy.deepcopy(penalties)
        memory[gen]["archive"] = copy.deepcopy(reproducers)
        memory[gen]["costs"] = copy.deepcopy(Ct)

        with open(global_var.output_memory_path, 'a', newline="") as file:
            file.write(f"t = {gen}\n")
            file.write("population:\n")
            for part in memory[gen]["pop"]:
                file.write(str(part)+"\n")
            file.write("\nselected:\n")
            for arch in memory[gen]["archive"]:
                file.write(str(arch[0])+"\n")
                file.write("costs: " + str(arch[1])+"\n")
                if penalty:
                    file.write("penalty: " + str(arch[2])+"\n")
            file.write("____________________________________________________________________________________________________________________________\n")
        

    return memory


# Problem to test the algorithm

# def create_part(n):
#     return [random.random() for _ in range(n)]

# pop = [create_part(2) for _ in range(50)]

# def cost1(part):
#     x, y = part
#     A1 = 0.5*np.sin(1)-2*np.cos(1)+np.sin(2)-1.5*np.cos(2)
#     A2 = 1.5*np.sin(1)-np.cos(1)+2*np.sin(2)-0.5*np.cos(2)
#     B1 = 0.5*np.sin(x)-2*np.cos(x)+np.sin(y)-1.5*np.cos(y)
#     B2 = 1.5*np.sin(x)-np.cos(x)+2*np.sin(y)-0.5*np.cos(y)
#     return 1+(A1-B1)**2+(A1-B2)**2, (x+3)**2+(y+1)**2

# def cost2(part):
#     x, y = part
#     return x, y 

# memory = MOGA(pop, 
#                400,
#                cost2,
#                weakly_dominate,
#                penalty=False,
#                crossover_method="unfair average",
#                degrees_of_freedom=[-1000,1000], 
#                mutation_method="non uniform"
#                )


# for k in range(len(memory)):
#     # plt.clf()
#     fuel = [el[0] for el in memory[k]["pop"]]
#     time = [el[1] for el in memory[k]["pop"]]
#     plt.plot(fuel, time, "b,")
#     plt.draw()
#     plt.pause(0.01)

# print(memory[len(memory)-1]["pop"])
# plt.show()
    