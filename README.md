# Optimisation of pumping rates of Wells in a given area with MODFLOW and Python
> Design of an algorithm to optimize the pumping rates of hundreds of wells in Ground Water models provided by GMS. 
> The optimisation take into consideration lot of concepts around the Ground-Water field such as the drawdown, the
> heads, the aquifers-river exchanges, etc. Therefore, the optimisation is not a single objective optimisation but
> a multi-objective optimisation. The purpose is to developped to optimise this multiobjective function in Python 
> and using the software MODFLOW to modelise the GW model.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Architecture](#architecture)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->


## General Information
- Using MODFLOW and a GMS object to build multi-objective fitness function to optimise.
- Optimise the Multi-Objective function.
- Give flexible and accessible features to be changed in a menu. 


## Technologies Used
- Python - version 3.9.12
- Numpy - version 1.19.5
- Matplotlib - version 3.5.2
- MODFLOW - version 2000 or 2005
- GMS - version 10.4.7

### Architecture

    .
    ├── wells_opti                                  # Project folder
      ├── data                                      # Folder containing the models and the results of the optimisations
      ├── image                                     # Image storing
      ├── scripts                                   # Python scripts folder

    .
    ├── data                                        # Close up on the data folder
      ├── optimisationx                             # Folder containing the results of the optimisations
        ├── model_name_output                       # Folder containing the outputs of the optimisations
        ├── model_name_input_variables.json         # Json file containing the input variables
  
> In the following We will detail the files inside an "optimisationx" folder and in the scripts folder.

## Optimisationx folder
    .
    ├── optimisationx                                             # Folder containing the results of the optimisations
        ├── model_name_output
            ├── cluster.csv                                       # Each row is a cluster and it contains the cell id of the Well
            ├── mopso_costs-N_iter-N_pop_model_name.csv           # Each row is a generation and each column is a particles, it contains the multi-cost
            ├── mopso_pareto_opt-N_iter-N_pop_model_name.csv      # Each row is a generation and each column is a pareto optimum, it contains the multi-cost
            ├── mopso_penalties_opt-N_iter-N_pop_model_name.csv   # Each row is a generation and each column is a pareto optimum, it contains the penalty
            ├── mopso_memory-N_iter-N_pop_model_name.txt          # .txt file of the history of the optimisation. Decision variables of the particles can be found here.
        ├── model_name_input_variables.json                       # Json file containing the input variables

## Scripts folder
    .
    ├── scripts                                    # Folder containing the Python scripts
        ├── data_management_tools.py               # Reads the output data from GMS and computes the fitness function
        ├── display_data.py                        # Displays the output of the optimisation and the map (River + Wells + Areas)
        ├── global_var.py                          # Sets up all the default variables (they are then modified by main.py) 
        ├── GMS_objects.py                         # Contains the classes of objects from GMS (River, Well, Cell) and Area = set of Wells
        ├── main_MOopti.py                         # Puts the input data in the optimisation algorithms and runs them
        ├── main.py                                # Display the API which will collect the input data
        ├── modules.py                             # Imports the packages
        ├── moga_tools.py                          # Multi-Objective Genetic Algorithm
        ├── mopso_tools.py                         # Multi-Objective Particle Swarm Optimisation
        ├── wells_clustering.py                    # Creates the Area by performing a chosen clustering method on the set of Wells
        ├── README.md                              # This file which explain how to use the code
                                                     

> The main files which are suceptible to be modified are the following: data_management_tools.py (to modify the fitness function or the penalty), mopso/moga_tools.py (to improve the optimisation algorithms), wells_clustering.py (to add a new clustering method or change the clustering parameters).

## Screenshots
![Example screenshot](./img/screenshot.png)
<!-- If you have screenshots you'd like to share, include them here. -->


## Setup
This project will work with MODFLOW 2000 or 2005 and GMS 10.4.7. We cannot ensure the well behaving of the code with other version because many files from MODFLOW and GMS are encrypted in bytes under a special format, and We are reading them manually and according to this format. If the structure of the binary files, output from MODFLOW and GMS came to a change It can deeply affect the code functionning.

Furthermore, It is required to keep the exact same architecture as It is described here. Otherwise, the different paths and pointers in the code will fail. 


## Usage
How does one go about using it?
Provide various use cases and code examples here.

`write-your-code-here`


## Project Status
Project is: _in progress_ / _complete_ / _no longer being worked on_. If you are no longer working on it, provide reasons why.


## Room for Improvement
Include areas you believe need improvement / could be improved. Also add TODOs for future development.

Room for improvement:
- Improvement to be done 1
- Improvement to be done 2

To do:
- Feature to be added 1
- Feature to be added 2


## Acknowledgements
Give credit here.
- This project was inspired by...
- This project was based on [this tutorial](https://www.example.com).
- Many thanks to...


## Contact
Created by [@flynerdpl](https://www.flynerd.pl/) - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->


