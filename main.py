"""
Author Lilian Bosc
This file contains the code which displays the form with the library tkinter, recover the data from it
and modify the variables in the file global_var.py
"""

from tkinter import *

from modules import *
import global_var
from data_management_tools import *
from mopso_tools import *

from main_MOopti import *

fields = (
          'file name', 
          'number of rows', 
          'number of columns', 
          'number of iterations', 
          'population size',
          'archive size',
          'w',
          'c1',
          'c2',
          'turbulence proba',
          'threshold',
          'initial population lb',
          'initial population ub',
          'Qlb',
          'Qub',
          'Cmodel',
          'clustering_algo',
          )


def get_variables_json(entries):
    variables = {}
    filename_ = (str(entries['file name'].get()))
    variables["filename"] = filename_
    max_row_ = (int(entries['number of rows'].get()))
    variables["max_row"] = max_row_
    max_col_ = (int(entries['number of columns'].get()))
    variables["max_col"] = max_col_
    N_iter_ = (int(entries['number of iterations'].get()))
    variables["N_iter"] = N_iter_
    N_pop_ = (int(entries['population size'].get()))
    variables["N_pop"] = N_pop_
    archive_size = (str(entries['archive size'].get()))
    if archive_size == "auto":
        variables["archive_size"] = archive_size
    else:
        variables["archive_size"] = int(archive_size)
    w_ = (float(entries['w'].get()))
    variables["w"] = w_
    c1_ = (float(entries['c1'].get()))
    variables["c1"] = c1_
    c2_ = (float(entries['c2'].get()))
    variables["c2"] = c2_
    p_turbulence_ = (float(entries['turbulence proba'].get()))
    variables["p_turbulence"] = p_turbulence_
    threshold_ = (float(entries['threshold'].get()))
    variables["threshold"] =  threshold_
    init_lb_ = (float(entries['initial population lb'].get()))
    init_ub_ = (float(entries['initial population ub'].get()))
    variables["init_range"] = [init_lb_, init_ub_]
    Qub_ = (float(entries['Qub'].get()))
    variables["Qub"] = Qub_
    Qlb_ = (float(entries['Qlb'].get()))
    variables["Qlb"] = Qlb_
    Qub_ = (float(entries['Qub'].get()))
    variables["Qub"] = Qub_
    Cmodel_ = (float(entries['Cmodel'].get()))
    variables["Cmodel"] = Cmodel_
    display = (bool(entries['display'].get()))
    variables["display"] = display
    optimisation_method = (str(entries['optimisation_method'].get()))
    variables["optimisation_method"] = optimisation_method
    clustering_algo = (str(entries['clustering_algo'].get()))
    variables["clustering_algo"] = clustering_algo

    go_to_the_root()
    optimisations = []
    for el in os.listdir('data'):
        if "optimisation" in el:
            optimisations.append(el)

    global_var.filename = filename_
    global_var.opti_folder = "optimisation"+str(len(optimisations)+1)
    global_var.output_folder_path = "data/"+global_var.opti_folder+"/"+variables["filename"]+"_output"
    os.mkdir("data/"+global_var.opti_folder)
    os.makedirs(global_var.output_folder_path)
    
    with open("data/"+global_var.opti_folder+"/"+filename_+'_input_variables.json', 'w') as file:
        json.dump(variables, file)
    

def makeform(root, fields):
    entries = {}
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=22, text=field+": ", anchor='w')
        if field == "file name":
            ent = Entry(row)
            ent.insert(0,"modelain2.0")
        elif field == "number of rows":
            ent = Entry(row)
            ent.insert(0,"164")
        elif field == "number of columns":
            ent = Entry(row)
            ent.insert(0,"118")
        elif field == "number of iterations":
            ent = Entry(row)
            ent.insert(0,"600")
        elif field == "population size":
            ent = Entry(row)
            ent.insert(0,"20")
        elif field == "archive size":
            ent = Entry(row)
            ent.insert(0,"auto")
        elif field == "w":
            ent = Entry(row)
            ent.insert(0,"0.7")
        elif field == "c1" or field == "c2":
            ent = Entry(row)
            ent.insert(0,"2.0")
        elif field == "Cmodel":
            ent = Entry(row)
            ent.insert(0,"1000")
        elif field == "turbulence proba":
            ent = Entry(row)
            ent.insert(0,"0.1")
        elif field == "threshold":
            ent = Entry(row)
            ent.insert(0,"4.0")
        elif field == "initial population lb":
            ent = Entry(row)
            ent.insert(0,"-2500.0")
        elif field == "initial population ub":
            ent = Entry(row)
            ent.insert(0,"-500.0")
        elif field == "Qlb":
            ent = Entry(row)
            ent.insert(0,"-2500.0")
        elif field == "Qub":
            ent = Entry(row)
            ent.insert(0,"-500.0")
        row.pack(side = TOP, fill = X, padx = 5 , pady = 5)
        lab.pack(side = LEFT)
        ent.pack(side = RIGHT, expand = YES, fill = X)
        entries[field] = ent

    OptionList = ["MOPSO", "MOGA"]
    variable = StringVar()
    variable.set(OptionList[0])
    opt = OptionMenu(root, variable, *OptionList)
    opt.pack(side = LEFT, padx = 5, pady = 5)
    entries["optimisation_method"] = variable

    OptionList1 = [ 'ward', 
                    'kmeans', 
                    'kmedoids', 
                    'affinity propagation',
                    'mini batch kmeans',
                    'birch',
                    ]
    var1 = StringVar()
    var1.set(OptionList1[0])
    opt1 = OptionMenu(root, var1, *OptionList1)
    opt1.pack(side = LEFT, padx = 5, pady = 5)
    entries["clustering_algo"] = var1

    var = BooleanVar()
    Checkbutton(root, text="display output",padx = 5, variable=var).pack(side = LEFT, padx = 5, pady = 5)
    entries["display"] = var

    return entries

if __name__ == '__main__':
    root = Tk()
    ents = makeform(root, fields)
    root.bind('<Return>')

    b2 = Button(root, text="Launch Optimization", command=root.destroy).pack(side = RIGHT, padx = 6, pady = 5)

    b1 = Button(root, text = 'Apply',
      command=(lambda e = ents: get_variables_json(e)))
    b1.pack(side = RIGHT, padx = 6, pady = 5)

    
    root.mainloop()

    with open("data/"+global_var.opti_folder+"/"+global_var.filename+'_input_variables.json') as f:
        variables = json.load(f)

    ## setup
    global_var.max_row = variables["max_row"]
    global_var.max_col = variables["max_col"]

    # MOPSO variables
    global_var.N_iter = variables["N_iter"]
    global_var.N_pop = variables["N_pop"]
    global_var.archive_size = variables["archive_size"]
    global_var.w = variables["w"]
    global_var.c1 = variables["c1"]
    global_var.c2 = variables["c2"]
    global_var.p_turbulence = variables["p_turbulence"]

    global_var.threshold = variables["threshold"]

    # Discharges range
    global_var.Qlb = variables["Qlb"]
    global_var.Qub = variables["Qub"]

    # Init pop range
    global_var.init_range = variables["init_range"]

    global_var.Cmodel = variables["Cmodel"]
    global_var.clustering_algo = variables['clustering_algo']

    global_var.environment_file_path = "data/"+global_var.filename+"_MODFLOW/"+global_var.filename+".h5"
    global_var.heads_file_path = "data/"+global_var.filename+"_MODFLOW/"+global_var.filename+".hed"
    global_var.leakage_file_path = "data/"+global_var.filename+"_MODFLOW/"+global_var.filename+".ccf"
    global_var.drawdown_file_path = "data/"+global_var.filename+"_MODFLOW/"+global_var.filename+".drw"
    global_var.training_file_path = "data/"+global_var.filename+"_MODFLOW/training.h5"
    
    global_var.output_costs_path = global_var.output_folder_path + f"/mopso_costs-{global_var.N_iter}-{global_var.N_pop}_{global_var.filename}.csv"
    global_var.output_memory_path = global_var.output_folder_path + f"/mopso_memory-{global_var.N_iter}-{global_var.N_pop}_{global_var.filename}.txt"
    global_var.output_pareto_opt_path = global_var.output_folder_path + f"/mopso_pareto_opt-{global_var.N_iter}-{global_var.N_pop}_{global_var.filename}.csv"
    global_var.output_penalties_path = global_var.output_folder_path + f"/mopso_penalties_opt-{global_var.N_iter}-{global_var.N_pop}_{global_var.filename}.csv"
    global_var.clusters_file_path = global_var.output_folder_path + "/clusters.csv"
    
    optimize(optimisation_method=variables["optimisation_method"], clustering_algo=variables['clustering_algo'], display=variables["display"])