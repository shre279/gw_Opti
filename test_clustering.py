from tkinter import *
import csv
from modules import *
import global_var
from data_management_tools import *
from mopso_tools import *
from colorspacious import cspace_converter
from main_MOopti import *
import global_var
from sklearn.neighbors import NearestNeighbors
import sklearn.cluster as clust
from sklearn_extra.cluster import KMedoids
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import numpy as np

wells, areas, river = read_cells_clustering(algo="ward", dim=2)
#print(wells, areas, river)
clusters, labels= cluster_wells(wells, river, 'ward', dim=2, d_river=False, MinPts=4, eps=3.5, n_clusters=40)
print("Now printing the clusters and labels")
print(clusters, labels)
display_clusters(clusters, labels, river)
plt.plot(markersize=20)
# filename = 'output_points.csv'
# Open the CSV file in write mode
# with open(filename, 'w', newline='') as file:

#     # Create a CSV writer object
#         writer = csv.writer(file)

#     # Write the header row
#         writer.writerow(['Cluster', 'label'])
#    # print the csv in required data format
#         for x in range(len(clusters)):
              
#                 temp = clusters[x]
#                 temp3 = []
#                 for clus in temp:
#                         temp3.append(clus[0]+[x+1])
#                 temp2 = [x+1]*len(clusters[x])
#                 print(len(clusters[x]))
#                 print(len(temp3))
#                 temp.append(temp2)
#                 writer.writerows(temp3)

    # Write the values of the variables as a row in the CSV file
        #writer.writerows([clusters, labels])
