"""
Author Lilian Bosc 
Latest update: 13/12/2022

The Classes which will prove usefull to manipulate objects extracted from GMS such as a Cell of the grid, a Well, a River, etc.

commented
"""


import global_var
import numpy as np

def id_to_position(id_):
    """
    converts the id (id_) of a cell in the grid into its coordinates according to max_col, max_row from global_var
    """
    n = global_var.max_row*global_var.max_col
    cell_layer = (id_-1)//n + 1 # should still work with more than one layer
    cell_row = (id_ - (cell_layer-1))//global_var.max_col + 1
    cell_col = id_ - (cell_layer-1)*n - (cell_row-1)*global_var.max_col
    return [cell_row, cell_col, cell_layer]

def distance_in_cell(cell1, cell2):
    """
    To manage with layers We will use the 3D euclidian distance
    """
    return ((cell1.position[0]-cell2.position[0])**2+(cell1.position[1]-cell2.position[1])**2+(cell1.position[2]-cell2.position[2])**2)**0.5

class Cell:
    """
    A cell from the GMS Grid
    """
    def __init__(self, id, storage_id):
        self.id = id
        self.storage_id = storage_id
        self.position = id_to_position(id)

    def __repr__(self):
        return f"cell {self.id}"

class Well(Cell):
    """
    Inherit from Cell but it is a Well with all its attributes from GMS
    """
    def __init__(self, id, storage_id, wells_number, commune=None, HK1=None, SY1=None, start_hd=None, bot1=None, top1=None, ET=None, recharge=None, discharges=None, drawdowns=None, area=None, ):
        super().__init__(id, storage_id)
        self.commune = commune
        self.wells_number = wells_number
        self.discharges = discharges
        self.drawdowns = drawdowns
        self.area = area
        self.HK1 = HK1
        self.SY1 = SY1
        self.start_hd = start_hd
        self.bot1 = bot1
        self.top1 = top1
        self.ET = ET
        self.recharge = recharge

    def __repr__(self):
        return f"Well at the cell {self.id}, and {self.commune}"

class River(Cell):
    """
    Inherit from Cell but it is a River cell. Will prove useful to compute the distance of a Well from the whole River
    """
    def __init__(self, id, storage_id, river_name=None):
        super().__init__(id, storage_id)
        self.river_name = river_name

class Area:
    """
    Area object is a set of wells to which We will give the same discharge. There will be as much areas as there is clusters of Wells
    """
    def __init__(self, id, name, discharge=0, id2=None):
        self.id = id
        self.id2 = id2 # id2 will be the index of the area in the particle
        self.name = name
        self.wells = []
        self.discharge = discharge
        self.total_discharge = len(self.wells)*discharge

    def __repr__(self):
        return self.name

    def update(self):
        self.total_discharge = len(self.wells)*self.discharge

    def find_mean(self):
        matrix = [well.discharges for well in self.wells]
        mean = np.mean(matrix, axis=0)
        return sum(mean)

    def find_center(self):
        wells = self.wells
        n = len(wells)
        dist_matrix = [[0 for j in range(n)] for i in range(n)]
        # fill the symetric matrix
        for i in range(n):
            for j in range(n):
                if i < j:
                    dist_matrix[i][j] = distance_in_cell(wells[i], wells[j])
                elif i > j:
                    dist_matrix[i][j] = dist_matrix[j][i]

        # finding the minimum of the sum of all distances
        dist_list = [sum(row) for row in dist_matrix]
        mini = min(dist_list)

        # The index of the min is the index of the median well
        index = dist_list.index(mini)
        return wells[index]

# print("GMS_object")