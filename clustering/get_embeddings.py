import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import sys
import numpy as np
from sklearn.cluster import KMeans
from k_means import K_Means
import pandas as pd

def get_cluster_labels(labels, names):

    # Create dictionary
    label_name = {}

    for i in range(10):
            
        label_name[i] = []

    for j in range(len(labels)):

        label_name[labels[j]].append(names[j])

    return label_name

def output_cluster(label_name):

    cluster_output = pd.DataFrame(list(label_name.items()))

    cluster_output.to_csv("cluster_output.csv", index=False, header=False, sep="\t")


embeddings = np.round(np.load("embeddings/embeddings.npy"), decimals=6)
labels = np.load("embeddings/labels.npy")
label_strings = np.load("embeddings/label_strings.npy")


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

print(model.centroids)
print("\n")
print(model.labels)
print("\n")
# print(model.label_feature)

label_names = get_cluster_labels(model.labels, names)
output_cluster(label_names)




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
