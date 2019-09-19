import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import sys
import numpy as np
from sklearn.cluster import KMeans
from k_means import K_Means
import pandas as pd
import csv
from collections import Counter
import os

#TODO: move function to k_means
# Separate file for evaluate
# Run experiment 1

def get_cluster_labels(labels, names):

    # Create dictionary
    label_name = {}

    for i in range(10): 
        label_name[i] = []

    for j in range(len(labels)):
        label_name[labels[j]].append(names[j])

    label_name_sorted = sorted(label_name.items() ,  key=lambda x: x[1])

    clusters = []

    for i in range(len(label_name_sorted)):
        clusters.append(label_name_sorted[i][1])

    return clusters


embeddings = np.round(np.load("embeddings/embeddings.npy"), decimals=6)
labels = np.load("embeddings/labels.npy")
label_strings = np.load("embeddings/label_strings.npy")

#print(labels)


# Clustering
X = np.round(embeddings[:-1], decimals=6)
# label: names
names = label_strings[:-1]

# KMeans from sklearn
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
# print(kmeans.labels_)
# print("\n")
# print(kmeans.cluster_centers_)
# print("\n")
# y = embeddings[-1].reshape(1, -1)
# p = kmeans.predict(y)
# print(p)

# KMeans implementation
model = K_Means(num_clusters=10)
model.fit(X)

# print(model.centroids)
# print("\n")
# print(model.labels)
# print("\n")
# print(model.label_feature)

label_names = get_cluster_labels(model.labels, names)

print(label_names)

# freq_names = []

# for i in range(len(label_names)):

#     c = Counter(label_names[i])
#     freq_names.append(c.most_common(1))

# sorted_freq_names = sorted(freq_names)

# remove_duplicate = np.unique(sorted_freq_names, axis=0)

# print("\n")
# print(sorted_freq_names)
print("\n")

files=[]
files = [f for f in sorted(os.listdir("train/"))]

num_faces = [len(os.listdir("train/{}".format(i))) for i in files]

true_label = []

for i in range(len(files)):
    true_label.append((files[i], num_faces[i]))


print("\n")
print(true_label)
print("\n")

# matching labels in same cluster
true_positive = []
# non matching labels in same clusters
false_positive = []
# non matching labels in different clusters
false_negative = []

for i in range(len(label_names)):

    true_positive.append(label_names[i].count(true_label[i][0]))

    # size of cluster - true positive
    false_positive.append(len(label_names[i]) - label_names[i].count(true_label[i][0]))

    # size of true label - true positive
    false_negative.append(true_label[i][1] - label_names[i].count(true_label[i][0]))

tp = sum(true_positive)
fp = sum(false_positive)
fn = sum(false_negative)

precision = tp/(tp+fp)
recall = tp/(tp+fn)
f_measure = 2*((precision*recall)/(precision+recall))

print(precision)
print(recall)
print(f_measure)

print("\n")


with open("cluster_output.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(label_names)





########### Prediction ###################

# print("\n")
# print(model.predict(embeddings[-1].reshape(1, -1)))

########### Plotting ####################
#plt.scatter(X[:,0], X[:,1], s=150)

#colors of plot
# colors = 10*["g","r","c","b","k"]

# #plot centroids
# for k in range(model.num_clusters):

#     plt.scatter(model.centroids[k][0], model.centroids[k][1], marker="o", color="k", s=30, linewidths=5)

# for i in model.labels:
#   color = colors[i]
#   for feature in X[model.labels == i]:
#       plt.scatter(feature[0], feature[1], marker="x", color=color, s=30, linewidths=5)

# plt.show()


# Predict new datapoint
##unknowns = np.array([[1,3],
##                     [8,9],
##                     [0,3],
##                     [5,4],
##                     [6,4],])
##
##for unknown in unknowns:
##    classification = model.predict(unknown)
##    plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)
##

#plt.show()
