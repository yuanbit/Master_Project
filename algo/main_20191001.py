#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import csv
from collections import Counter
import matplotlib.cm as cm
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from itertools import chain, combinations
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
import itertools
import time
from sklearn.cluster import AffinityPropagation
import re
import pickle

def get_labels_idx(keys, labels_string):
    # key: category
    # value: tuple of name and index, e.g. ('Andy Murray', 0)
    labels = {k: [] for k in keys}

    for i in range(len(labels_string)):
        for k in keys:
            if k in labels_string[i]:
                name = labels_string[i].replace(k, "")
                labels[k].append(i)
    return labels

def get_clusters_dict(labels):
    # key: label
    # value: indices of images
    clusters = {}

    for idx, label in enumerate(labels):
        if label not in clusters:
            # The label is seen for first time, create a new list.
            clusters[label] = [idx]
        else:
            clusters[label].append(idx)
            
    return clusters

# Create label pairs

def create_label_pairs(labels):
    
    label_pairs = {}
    
    for key, value in labels.items():
        label_pairs[key] = list(itertools.combinations(value, 2)) 
        
    label_pairs_concat = []

    for key, value in label_pairs.items():
        label_pairs_concat += value
        
    return label_pairs_concat

# F-measure

def f_measure(true_labels, cluster_labels, algo):
    
    true_positive = list(set(true_labels).intersection(cluster_labels))
    false_positive = list(set(cluster_labels) - set(true_labels))
    false_negative = list(set(true_labels) - set(cluster_labels))

    TP = len(true_positive)
    FP = len(false_positive)
    FN = len(false_negative)
    
    precision = round(TP/(TP+FP), 2)
    
    recall = round(TP/(TP+FN), 2)
    
    f_measure = round(2*((precision*recall)/(precision+recall)), 2)
    
    print("{} F-Measure: {}".format(algo, f_measure))
    print("{} Precision: {}".format(algo, precision))
    print("{} Recall: {}".format(algo, recall))
    print("{} Number of False Positives: {}".format(algo, FP))


# In[40]:


e = np.load("data1_embeddings.npy")
n = np.load("data1_names.npy")

print(len(e))


# In[ ]:


# read in embeddings from Openface

data = list(csv.reader(open("embeddings/data1_mtcnn_160_embeddings/reps.csv")))
label_s = list(csv.reader(open("embeddings/data1_mtcnn_160_embeddings/labels.csv")))

openface_embeddings = np.asarray(data, dtype=float)
openface_raw_labels = []

for i in range(len(label_s)):
    openface_raw_labels.append(label_s[i][1])

openface_raw_labels = [re.sub("./datasets/data1_mtcnnpy_160/", "", x) for x in openface_raw_labels]
openface_raw_labels = [re.sub("(?=\/).*$", "", x) for x in openface_raw_labels]

print(openface_embeddings.shape)
print(openface_raw_labels[0])


# In[29]:


from collections import Counter

a = [item for item, count in Counter(dlib_raw_labels).items() if count > 1]


# In[30]:


print(a)


# In[32]:



for j in range(len(a)):
    indices = [i for i, x in enumerate(dlib_raw_labels) if x == a[j]]
    
    print(indices)
    
d = [301, 334, 382, 407, 443, 869, 933, 944, 949, 964, 1668, 1720]

print(len(dlib_raw_labels))


# In[ ]:


# read in embeddings from FaceNet",

facenet_embeddings = np.load("embeddings/data1/embeddings.npy")
t_labels = np.load("embeddings/data1/labels.npy")
label_strings = np.load("embeddings/data1/label_strings.npy")

encoding = 'utf-8'

# decode from byte to string
l = [str(x, encoding) for x in label_strings]
label_decoded = [x.replace('_', ' ') for x in l]

print(facenet_embeddings.shape)


# In[ ]:


## Starting clustering and evaluation

keys = ["tennis", "basketball", "golf", "fighter", "soccer"]

# Get label/index dictionary
facenet_labels = get_labels_idx(keys, label_decoded)
openface_labels = get_labels_idx(keys, openface_raw_labels)
dlib_labels = get_labels_idx(keys, dlib_raw_labels)

for k, v in openface_labels.items():
    print(k)
    print(len(v))

## Choose method
feature_extraction_method = "dlib"

if feature_extraction_method == "openface":

    X = openface_embeddings
    # Create ground truth pairs for evaulation
    true_label_pairs = create_label_pairs(openface_labels)
    
elif feature_extraction_method == "facenet":
    
    X = facenet_embeddings
    # Create ground truth pairs for evaulation
    true_label_pairs = create_label_pairs(facenet_labels)

elif feature_extraction_method == "dlib":
    
    X = dlib_embeddings
    # Create ground truth pairs for evaulation
    true_label_pairs = create_label_pairs(dlib_labels)


# In[ ]:


# K-means 
num_clusters = 5

start_time = time.time()

kmeans = KMeans(n_clusters = num_clusters).fit(X)
#print(kmeans.labels_)

k_means_clusters = get_clusters_dict(kmeans.labels_)

# print(labels)
# print("\n")
# print(k_means_clusters)

kmeans_label_pairs = create_label_pairs(k_means_clusters)

#F-measure

f_measure(true_label_pairs, kmeans_label_pairs, "K-means")

print("--- %s seconds ---" % (time.time() - start_time))

print()

# Hierarchical Agglomerative Clustering

start_time = time.time()

clustering = AgglomerativeClustering(n_clusters=5, distance_threshold=None).fit(X)
hac_clusters = get_clusters_dict(clustering.labels_)

hac_label_pairs = create_label_pairs(hac_clusters)

f_measure(true_label_pairs, hac_label_pairs, "HAC")

print("--- %s seconds ---" % (time.time() - start_time))

print()

# DBSCAN

start_time = time.time()

clustering = DBSCAN(eps=1, min_samples= 3).fit(X)
DBSCAN_cluster = get_clusters_dict(clustering.labels_)

print(clustering.labels_)
print("\n")
print(len(DBSCAN_cluster))
print("\n")
DBSCAN_label_pairs = create_label_pairs(DBSCAN_cluster)

f_measure(true_label_pairs, DBSCAN_label_pairs, "DBSCAN")

print("--- %s seconds ---" % (time.time() - start_time))

print()

# Spectral Clustering

start_time = time.time()

clustering = SpectralClustering(n_clusters=5).fit(X)

spectral_cluster = get_clusters_dict(clustering.labels_)

spectral_label_pairs = create_label_pairs(spectral_cluster)

f_measure(true_label_pairs, spectral_label_pairs, "Spectral")

print("--- %s seconds ---" % (time.time() - start_time))

print()

# Gaussian Mixture EM

start_time = time.time()

gmm_labels = GaussianMixture(n_components=5, init_params='kmeans').fit_predict(X)

gmm_clusters = get_clusters_dict(gmm_labels)

gmm_label_pairs = create_label_pairs(gmm_clusters)

f_measure(true_label_pairs, gmm_label_pairs, "GMM")

print("--- %s seconds ---" % (time.time() - start_time))

print()

# Birch

start_time = time.time()

brc = Birch(n_clusters=5, threshold=0.58, compute_labels=True).fit(X) 

birch_labels = brc.predict(X)

birch_clusters = get_clusters_dict(birch_labels)

birch_label_pairs = create_label_pairs(birch_clusters)

f_measure(true_label_pairs, birch_label_pairs, "Birch")

print("--- %s seconds ---" % (time.time() - start_time))

print()

# Affinity Propagation
start_time = time.time()

clustering = AffinityPropagation().fit(X)

ap_clusters = get_clusters_dict(clustering.labels_)

print(len(ap_clusters))

ap_label_pairs = create_label_pairs(ap_clusters)

f_measure(true_label_pairs, ap_label_pairs, "Affinity Porpagation")

print("--- %s seconds ---" % (time.time() - start_time))

print()

# Mean shift

start_time = time.time()

clustering = MeanShift(bandwidth=1).fit(X)

mean_shift_cluster = get_clusters_dict(clustering.labels_)

print(clustering.labels_)
print("\n")
print(len(mean_shift_cluster))
print("\n")
mean_shift_label_pairs = create_label_pairs(mean_shift_cluster)

f_measure(true_label_pairs, mean_shift_label_pairs, "Mean Shift")

print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:


# Find error pairs

true_positive = list(set(true_label_pairs).intersection(hac_label_pairs))
false_positive = list(set(hac_label_pairs) - set(true_label_pairs))
false_negative = list(set(true_label_pairs) - set(hac_label_pairs))

print(false_negative)


# In[ ]:


print(label_decoded[27])
print(label_decoded[46])


# In[ ]:


# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(X)
# reduced_centroids = pca.fit_transform(kmeans.cluster_centers_)

# colors = ["#ffe119", "#f032e6", "#9A6324", "#3cb44b", "#e6194B", "#f58231", "#ffe119", "#469990", "#42d4f4", "#4363d8", "#911eb4"]

# # plt.scatter(X[:,0], X[:,1], s=5)

# for i in kmeans.labels_:
#     color = colors[i]
#     for feature in principalComponents[kmeans.labels_ == i]:
#         plt.scatter(feature[0], feature[1], marker="x", color=color, s=5, linewidths=5)
#     plt.scatter(reduced_centroids[i][0], reduced_centroids[i][1], marker="o", color=color, edgecolors='black',  s=30, linewidths=1)

# plt.show()


# In[ ]:




