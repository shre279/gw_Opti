"""
Author Lilian Bosc 
Latest update: 13/12/2022

Functions usefull to read information out of a GMS project and compute fitness functions for optimisation
according to "Optimization of Groundwater Pumping and River-Aquifer Exchanges for Management of Water Resources",
by Mayank Bajpai, Shreyansh Mishra, Shishir Gaur, Anurag Ohri, Hervé Piégay, Didier Graillot, 2021.

commented
"""
import global_var
from modules import *
from wells_clustering import *
import pandas as pd


# Unwanted wells
if global_var.filename == "modelain2.0":
    l_well = 651 # the index where we stop counting the wells
    cp = 2006.7012992711
elif global_var.filename == "ain_domain3.0":
    l_well = 536
    cp = 7854.72739146830
elif global_var.filename == "modelain4.1":
    l_well = 634
    cp = 994.066666344009
else:
    l_well = 708
    cp = 1000


# Used functions
def go_to_the_root():
    # This function return one level back in the folder architecture
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("../")
    
def translate_bytes(list_of_bytes):
    """
    This function simply convert bytes into char

    Parameters
    -----------
    list_of_bytes

    Output
    ------
    a string of the characters encoded by the bytes
    """
    s = ""
    for byte in list_of_bytes:
        s+=chr(byte)

    return s

def read_cells_clustering(algo="communes", dim=3):
    """
    This function will read the cells from the HDF5 file and return the wells and areas classified with 
    well_clustering.py functions instead of just reading the well names.
    
    Parameters
    ----------
     - algo <str> among "dbscan", "kmedoids", "kmeans", "ward", "affinity propagation", "mini batch kmeans", "birch", "OPTICS", "communes"
     - dim <int> the number of parameters on which do the clustering
    Output
    ------
     - wells <array> of <GMS_object.Well> up to date
     - areas <array> of <GMS_object.Area> up to date
     - river <array> of <GMS_object.River>
    """
    go_to_the_root()

    # Read variables in the HDF5 file (there is a way to accelerate it with threads)
    environment_file = tables.open_file(global_var.environment_file_path)

    # data collecting
    river_cell_ids = environment_file.root["Stream"]["02. Cell IDs"]
    cell_ids = environment_file.root["Well"]["02. Cell IDs"]
    bytes_list = environment_file.root["Well"]["03. Name"]

    discharges_list = environment_file.root["Well"]["07. Property"]
    HK1_list = environment_file.root["Arrays"]["HK1"]
    SY1_list = environment_file.root["Arrays"]["SY1"]
    start_hd_list = environment_file.root["Arrays"]["StartHead1"]
    bot1_list = environment_file.root["Arrays"]["bot1"]
    top1_list = environment_file.root["Arrays"]["top1"]
    ET_list = [sum(periods)/len(periods) for periods in environment_file.root["ET"]["07. Property"][1]]
    recharge_list = [sum(periods)/len(periods) for periods in environment_file.root["Recharge"]["07. Property"][0]]

    # data treatment of the bytes' list, we need it to differenciate the wells from the arcs
    string = translate_bytes(bytes_list)
    list_of_objects = string.replace(" ","").split("\x00")[:-1] # list of all names (Arcs + Wells)
    
    # create objects
    river = [River(river_cell_ids[i], i, "Ain") for i in range(len(river_cell_ids))]

    wells = []
    for i in range(l_well):
        if "well" in list_of_objects[i]:
            id_ = cell_ids[i]
            well = Well(
                        id_,
                        i,
                        wells_number=len(wells),
                        discharges = discharges_list[0][i],
                        HK1 = HK1_list[id_],
                        SY1 = SY1_list[id_],
                        start_hd = start_hd_list[id_],
                        bot1 = bot1_list[id_],
                        top1 = top1_list[id_],
                        ET = ET_list[id_],
                        recharge = recharge_list[id_],
                        commune="commune"+list_of_objects[i].replace("well","") # We will save the attribute "wellx" as "communex" to avoid confusion
                        )
            wells.append(well)
    
    # Clustering all the wells with the method given by the algo parameter
    # (see wells_clustering.cluster_wells to learn more)
    clusters, labels = cluster_wells(wells, river, algo, dim=dim, n_clusters=17, d_river=True)
    n_clusters = len(set(labels))
    areas_dico = {}
    id2 = 0
    for key in set(labels):
        areas_dico[key] = Area(key, "area "+str(key), id2=id2)
        id2 += 1
    for i in range(len(wells)):
        areas_dico[labels[i]].wells.append(wells[i])
    areas = [areas_dico[key] for key in areas_dico.keys()]
    for i, area in enumerate(areas):
        area.discharge = area.wells[0].discharges[0] # as long as we are not dealing with stress periods
        area.update()
    environment_file.close()

    return wells, areas, river


def write_wells_property(array, areas):
    """
    This function write in the HDF5 file the property of each well according to the areas clustering.

    Parameters
    ----------
     - array Array(1 x n_areas) contains the total discharges of all areas
     - areas Array of <GMS_objects.Area> is the output[1] of read_cells() or read_cells_clustering()
    """
    go_to_the_root()
    hf5 = tables.open_file(global_var.environment_file_path, "r+")
    # hf5 = tables.open_file(training_file_path, "r+")
    n_stress_period = len(hf5.root["Well"]["07. Property"][0,:,:][0])
    properties = hf5.root["Well"]["07. Property"][0]
    for i in range(len(array)):
        discharge = array[i]
        discharges = [discharge for _ in range(n_stress_period)]
        area = areas[i]
        area.discharge = discharge
        area.update()
        for well in area.wells:
            properties[well.storage_id] = discharges
            well.discharges = discharges
    hf5.root["Well"]["07. Property"][0] = properties
    hf5.close()
    
def mf2k_h5_player():
    """This function run the mf2K.exe with the correct project."""
    go_to_the_root()
    os.chdir(f"data/{global_var.filename}_MODFLOW")
    filename_ = global_var.filename+".mfn"
    os.system(f"mf2k_h5.exe {filename_}")
    go_to_the_root()

def mf2k5_h5_player():
    """This function run the mf2K5.exe with the correct project."""
    go_to_the_root()
    os.chdir(f"data/{global_var.filename}_MODFLOW")
    filename_ = global_var.filename+".mfn"
    os.system(f"mf2k5_h5.exe {filename_}")
    go_to_the_root()

def ccf_find_row_col(filecontent):
    """
    This function find the occurrences of max_col and max_row in the ccf file. 
    From this position we can deduce the rest of the file encoding.
    Parameter
    ---------
    filecontent is a string of bytes

    Output
    ------
    list_index a list of all indexes where max_row 0 0 0 max_col is enconter
    """
    list_index = []
    for i in range(len(filecontent)):
        if filecontent[i]==global_var.max_col and filecontent[i+4]==global_var.max_row and filecontent[i+3] == filecontent[i+2] == filecontent[i+1] == 0:
            list_index.append(i)
    return list_index

def drw_find_values(filecontent):
    """
    This function find the chain of bytes b'    DRAWDOWN', the chain after which the file is begining.
    """
    list_index = []
    for i in range(len(filecontent)):
        if filecontent[i:i+12] == b'    DRAWDOWN':
            list_index.append(i)
    return list_index


def read_modflow_array(grid_bytes, nrow=global_var.max_row, ncol=global_var.max_col, nlay=global_var.glo_n_layers):
    """
    This function read an array produce by MODFLOW in bytes
    """
    if len(grid_bytes)/4 != nrow*ncol*nlay:
        print(len(grid_bytes)/4)
        print(global_var.filename)
        raise Exception(f"{global_var.filename}The number of cells in the grid does not match with the number bytes detected.")
    grid = []
    i = 0
    while i < len(grid_bytes):
        grid.append(struct.unpack('f', grid_bytes[i:i+4])[0]) # unpacking each pack of bytes and adding it to the array called grid 
        i += 4 # It is encoding on 4 bytes
    return grid

def read_ccf_stream_leakage():
    """
    This function will read the stream leakage out of a .ccf file.
    MODFLOW 2000
    MODFLOW 2005
    """
    go_to_the_root()
    print(global_var.leakage_file_path)
    file = open(global_var.leakage_file_path, 'rb')
    filecontent = file.read()
    file.close()
    i = ccf_find_row_col(filecontent)[-1] - 24
    CBC = {
            "ksp"  : struct.unpack("i", filecontent[i:i+4])[0],
            "kper" : struct.unpack("i", filecontent[i+4:i+8])[0],
            "desc" : str(filecontent[i+8:i+24]),
            "ncol" : struct.unpack("i", filecontent[i+24:i+28])[0],
            "nrow" : struct.unpack('i', filecontent[i+28:i+32])[0],
            "nlay" : struct.unpack('i', filecontent[i+32:i+36])[0],
            }
    if CBC["nrow"] != global_var.max_row or CBC["ncol"] != global_var.max_col:
        # If this Exception is not raised, then We can be sure that the file is being read correctly
        raise Exception("Row and col does not match between the model and the binary file.")
    i += 36 # startinh the reading at the begining of the grid (9 characters after)
    grid_bytes = filecontent[i:]
    grid = read_modflow_array(grid_bytes, CBC["nrow"], CBC["ncol"], CBC["nlay"])
    grid = np.reshape(grid, (global_var.max_row, global_var.max_col)) # making an array of size (row, col)
    return grid

def read_drawdown():
    """
    This function will read the drawdown out of a .drw file.
    MODFLOW 2000
    MODFLOW 2005
    """
    go_to_the_root()
    file = open(global_var.drawdown_file_path, 'rb')
    filecontent = file.read()
    file.close()
    drw_stress_period = []
    indexes = drw_find_values(filecontent)
    for i in indexes:
        array = read_modflow_array(filecontent[i+24: i+24+4*global_var.max_col*global_var.max_row])
        array = np.reshape(array, (global_var.max_row, global_var.max_col))
        drw_stress_period.append(array)

    return drw_stress_period


def generate_pop(n_areas, N_pop, Range=[global_var.Qlb, global_var.Qub]):
    """
    We will generate a population of p = [Q1, ..., Qn] where n will be the number of areas
    Particles will be identified by their indexes [p0, p1, ..., pN_pop-1]
    Parameters
    ----------
    n_areas (int)
    N_pop (int)
    Range (array) [Qlb, Qub]

    Output
    ------
    pop (Array N_pop x n_areas)
    """
    pop = [[random.uniform(Range[0], Range[1]) for j in range(n_areas)] for i in range(N_pop)]
    return pop

def total_discharge(areas):
    """
    Compute the total discharges of the model
    """
    nz = len(areas)
    S = 0
    for i in range(nz):
        area = areas[i]
        nzQi = area.total_discharge
        S += nzQi
    return S

def total_leakage():
    """
    Compute the total leakages River->Aquifer of the model
    """
    grid = read_ccf_stream_leakage()
    S = 0
    for row in grid:
        for el in row:
            if el < 0:
                S += el
    return S

def compute_penalty(wells):
    S = 0
    for well in wells:
        # print(well.drawdowns)
        di = well.drawdowns[-1]
        if di > global_var.dthreshold:
            # print(di)
            S += (di - global_var.dthreshold)**2
    ddist = S**0.5
    print("dd_distance =", ddist)
    print("Cmodel =", global_var.Cmodel)
    P = global_var.Cmodel*ddist
    return P


def mo_cost(wells, areas):
    """
    This function return the cost of a given modelised situation

    Parameters
    ----------
    - wells list of <GMS_object.Well> the output of read_cell() function 
    - areas list of <GMS_object.Area> the output of read_cell() function 

    Output
    ------
    - cost <float> the cost of the simulation (to maximise)
    - penalty <float> the penalty is usefull for the following of the optimization
    """
    n_stress_periods = len(areas[0].wells[0].discharges)
    # Reading drawdowns and filling the wells drawdowns
    drawdown_list = read_drawdown()
    for well in wells:
        drawdowns = []
        for i in range(len(drawdown_list)):
            drawdowns.append(drawdown_list[i][well.position[0]][well.position[1]])
        well.drawdowns = drawdowns

    # Computing the different caracteristics of the simulation
    total_disch = total_discharge(areas)
    total_leak = total_leakage()
    penalty = compute_penalty(wells)
    f1 = total_leak
    f2 = total_disch
    print("total leakage =", f1)
    print("total discharge =", f2)

    return f1, f2, penalty

def mo_get_costs(particle, penalty, areas, cost_function=mo_cost):
    """
    This function compute the costs of a particle with its penalty added

    Parameter
    ---------
    - particle
    - AREAS are the areas made in the begining

    Output
    ------
    - f1, f2
    - penalty
    """
    go_to_the_root()
    ## Setting up variables for the model
    # wells = read_wells()
    
    write_wells_property(particle, areas) # Writing in wells and updating the objects
    
    mf2k5_h5_player() # MODFLOW player
    
    wells, areas, river = read_cells_clustering() # Reading the values from the wells again

    f1, f2, penalty_ = cost_function(wells, areas) # compute the cost from the discharge of the wells, and the .ccf file and the .drw file generated by MODFLOW
    if penalty:
        return [f1, f2], penalty_
    else:
        return [f1, f2]

def str_line2list(line):
    # This function translate a string (of a list) into a real list 
    return [float(el) for el in line.replace("[", "").replace("]", "").replace(",", "").split(" ")]


# Testing
# wells, areas, river = read_cells_clustering()
# particle = [-1000 for _ in range(len(areas))]
# write_wells_property(particle, areas)
# wells, areas, river = read_cells_clustering()
# mf2k5_h5_player()
# print(read_drawdown()[0][50])