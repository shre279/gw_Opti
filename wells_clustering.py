"""
Author Lilian Bosc 
Latest update: 13/12/2022

This code clusters the Wells into Area using one of the different algorithms programmed

commented
"""
import matplotlib as mpl
from matplotlib import  transforms
import global_var
from sklearn.neighbors import NearestNeighbors
import sklearn.cluster as clust
from sklearn_extra.cluster import KMedoids
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from colorspacious import cspace_converter

def eucl_dist(cell1, cell2):
    """
    To manage with layers We will use the 3D euclidean distance
    """
    return ((cell1[0]-cell2[0])**2+(cell1[1]-cell2[1])**2)**0.5

def cluster_wells(wells, river, mode, dim=2, d_river=False, MinPts=4, eps=3.5, n_clusters=40):
    """
    This code will perform the clustering method of our choice (mode) on the set of wells. Each well possesses 
    a set of attribute which is described in the cloud variable line xx, and the clustering method will be 
    performed on the dim first attribute of this list.

    Parameters
    ----------
    - wells <GMS_objects.Well> set of wells
    - river <GMS_objects.River> set of cells which forms the whole River
    - mode <string> among dbscan, kmedoids, kmeans, wards, affinity propagation, mini batch kmeans, birch, 
      OPTICS and communes
    - dim <int>
    - d_river <Boolean> determine if We apply the distance with the river clusetring.
    """

    x_river = [cell.position[0] for cell in river]
    y_river = [cell.position[1] for cell in river]

    x_wells = [well.position[0] for well in wells]
    y_wells = [well.position[1] for well in wells]

    cloud = [[well.position[0], well.position[1], well.start_hd, well.SY1,well.HK1, well.bot1, well.top1, well.ET, well.recharge][:dim] for well in wells]
    
    if mode == "dbscan":
        # # This part is used to find the value of espilon for DBSCAN (we found eps=4)
        # neighb = NearestNeighbors(n_neighbors=2) # creating an object of the NearestNeighbors class
        # nbrs=neighb.fit(cloud) # fitting the data to the object
        # distances,indices=nbrs.kneighbors(cloud) # finding the nearest neighbours

        # # Sort and plot the distances results
        # distances = np.sort(distances, axis = 0) # sorting the distances
        # distances = distances[:, 1] # taking the second column of the sorted distances
        # plt.rcParams['figure.figsize'] = (5,3) # setting the figure size
        # plt.plot(distances) # plotting the distances
        # plt.show() # showing the plot

        MinPts = 4
        eps = 3.5
        res = clust.DBSCAN(eps = eps, min_samples = MinPts).fit(cloud) # fitting the model
        labels = res.labels_ # getting the labels        

    elif mode == "kmedoids":
        res = KMedoids(n_clusters=n_clusters, random_state=0).fit(cloud)
        labels = res.labels_

    elif mode == "kmeans":
        res = clust.KMeans(n_clusters=n_clusters, random_state=0).fit(cloud)
        labels = res.labels_
    
    elif mode == "ward":
        # # This part is here to display the ideal number of cluster
        # plt.figure(figsize=(10, 7))
        # dend = shc.dendrogram(shc.linkage(cloud, method='ward'))
        # plt.show()
        res = clust.AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward').fit(cloud)
        labels = res.labels_

    elif mode == "affinity propagation":
        res = clust.AffinityPropagation(random_state=0).fit(cloud)
        labels = res.labels_
        

    elif mode == "mini batch kmeans":
        res = clust.MiniBatchKMeans(n_clusters=n_clusters).fit(cloud)
        labels = res.labels_
        
    elif mode == "birch":
        res = clust.Birch(n_clusters=n_clusters).fit(cloud)
        labels = res.labels_

    elif mode == "OPTICS":
        res = clust.OPTICS().fit(cloud)
        labels = res.labels_

    elif mode == "communes":
        # if the mode is commune, then We will just separate the wells regarding their communes 
        # (see the function data_management.read_wells to learn more)
        labels = [-1 for _ in range(len(wells))]
        dico = {}
        for i, well in enumerate(wells):
            if well.commune not in dico.keys():
                dico[well.commune] = len(dico)
                labels[i] = dico[well.commune]
            else:
                labels[i] = dico[well.commune]
        labels = np.array(labels)
    
    else:
        raise ValueError("The mode is not recognised. It should be dbscan, kmeans, kmedoids, ward, affinity propagation, mini batch kmeans, birch, OPTICS or communes.")


    n_clusters = max(labels)+1
    if -1 in labels:
        clusters = [[] for _ in range(n_clusters+1)] # the noise will be stored in the final cluster
    else:
        clusters = [[] for _ in range(n_clusters)]
    for i in range(len(labels)):
        clusters[labels[i]].append((cloud[i], wells[i].storage_id))

    if d_river:
        # We will separate some clusters in two depending on their distance from the river
        n_clusters_ = n_clusters
        new_clusters = []
        for cluster in clusters:
            far = [] # the cells far from the river
            prox = [] # the cells close to the river
            for cell in cluster:
                dist = min([eucl_dist(cell[0], river_cell.position) for river_cell in river])
                if dist >= global_var.threshold:
                    far.append(cell)
                else:
                    prox.append(cell)
            if far != [] and prox != []:
                # if far and prox are not empty It means that We should separate the clusters 
                new_clusters.append(far)
                new_clusters.append(prox)
                for cell in far:
                    labels[cell[1]-332] = n_clusters_+1 # value 332 to change... No need to use labels
                n_clusters_ += 1
            else:
                new_clusters.append(cluster)
        return new_clusters, labels
                
    return clusters, labels

def display_clusters(clusters, labels, river):
    x_river = [cell.position[0] for cell in river]
    y_river = [cell.position[1] for cell in river]
    
    plt.figure(figsize=[5,8])
    # a=('#0F0F0F','#0F0F0F0F')
    for cluster in clusters[:-1]:
        x = [cluster[i][0][0] for i in range(len(cluster))]
        y = [cluster[i][0][1] for i in range(len(cluster))]
        base = plt.gca().transData
        rot = transforms.Affine2D().rotate_deg(280)
        
        plt.plot(x,y, 'X', transform= rot + base)

    x = [clusters[-1][i][0][0] for i in range(len(clusters[-1]))]
    y = [clusters[-1][i][0][1] for i in range(len(clusters[-1]))]
    if -1 in labels:
        
        plt.plot(x,y, 'rx', transform= rot + base)
        plt.title(f"{len(clusters)-1} clusters")
    else:
        
        plt.plot(x,y,'X', transform= rot + base)
        plt.title(f"{len(clusters)} clusters")
       
    plt.plot(x_river, y_river, "b.", transform= rot + base, markersize = .5)
    plt.savefig('ward80.png', format='png', dpi = 600)
    plt.show()

def display_map(areas, river):
    x_river = [cell.position[0] for cell in river]
    y_river = [cell.position[1] for cell in river]

    for area in areas:
        x = [area.wells[i].position[0] for i in range(len(area.wells))]
        y = [area.wells[i].position[1] for i in range(len(area.wells))]

        plt.plot(x,y, 'X')

    plt.plot(x_river, y_river, "b,")

    plt.title(f"{len(areas)} Well Zones")
    plt.legend = True
    plt.show()




