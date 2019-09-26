import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import csv
from collections import Counter
from k_means import K_Means
import matplotlib.cm as cm


def plot_data(data):

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)

    plt.scatter(principalComponents[:,0], principalComponents[:,1], s=30)

    plt.show()

def plot_clusters(data, labels, centroids, k):

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    reduced_centroids = pca.fit_transform(centroids)

    # colors = []

    # for i in range(self.num_clusters):
    #     colors.append('#%06X' % np.random.randint(0, 0xFFFFFF))

    colors = ["#e6194B", "#f58231", "#ffe119", "#469990", "#3cb44b", "#42d4f4", "#4363d8", "#911eb4", "#f032e6", "#9A6324"]

    # for i in labels:
    #   color = colors[i]
    #   for feature in principalComponents[labels == i]:
    #       plt.scatter(feature[0], feature[1], marker="x", color=color, s=50, linewidths=5)

    #plot centroids
    for k in range(k):
        plt.scatter(reduced_centroids[k][0], reduced_centroids[k][1], marker="o", color="k", s=50, linewidths=5)
    plt.show()


embeddings = np.round(np.load("embeddings/data1/embeddings.npy"), decimals=6)
t_labels = np.load("embeddings/data1/labels.npy")
label_strings = np.load("embeddings/data1/label_strings.npy")

X = embeddings

# KMeans from sklearn
kmeans = KMeans(n_clusters=5).fit(X)
# print(kmeans.labels_)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
reduced_centroids = pca.fit_transform(kmeans.cluster_centers_)

colors = ["#e6194B", "#f58231", "#ffe119", "#469990", "#3cb44b", "#42d4f4", "#4363d8", "#911eb4", "#f032e6", "#9A6324"]

# plt.scatter(X[:,0], X[:,1], s=5)

for i in kmeans.labels_:
  color = colors[i]
  for feature in principalComponents[kmeans.labels_ == i]:
      plt.scatter(feature[0], feature[1], marker="x", color=color, s=5, linewidths=5)
  plt.scatter(reduced_centroids[i][0], reduced_centroids[i][1], marker="o", color=color, edgecolors='black',  s=30, linewidths=1)

plt.show()


# plot_clusters(X, kmeans.labels_, kmeans.cluster_centers_, 5)

# print("\n")
# print(kmeans.cluster_centers_)
# print("\n")
# y = embeddings[-1].reshape(1, -1)
# p = kmeans.predict(y)
# print(p)

# 3 classes and 15 images
# subset_embeddings = embeddings[:15]
# subset_labels = t_labels[:15]

# Clustering
# X = np.round(embeddings[:-1], decimals=6)
# #X = subset_embeddings
# # label: names
# names = label_strings[:-1]
#
# # KMeans from sklearn
# kmeans = KMeans(n_clusters=10).fit(X)
# print(kmeans.labels_)
# # print("\n")
# # print(kmeans.cluster_centers_)
# # print("\n")
# # y = embeddings[-1].reshape(1, -1)
# # p = kmeans.predict(y)
# # print(p)
#
# # KMeans implementation
# model1 = K_Means(num_clusters=10)
# model1.fit(X)
#
# model2 = K_Means(num_clusters=10, k_means_plus_plus=True)
# model2.fit(X)

#print(model.centroids)
# print("\n")
# print(model1.labels)
#
# print(model2.labels)
# print("\n")
# print(model.label_feature)

# label_names = model.get_cluster_labels(model.labels, names)

# # output cluster results in csv file
# with open("cluster_output2.csv","w+") as my_csv:
#     csvWriter = csv.writer(my_csv, delimiter=',')
#     csvWriter.writerows(label_names)

# ####### Evaluation of Clusters ###################
# ev = Evaluate(label_names)

# # True lables with number of images for each label
# true_labels = ev.get_true_labels("datasets/test/")

#print(true_labels)

# Evaluation metric

# precsion, recall, f_measure = ev.compute_metric()

# print(precsion)
# print(recall)
# print(f_measure)

########### Prediction ###################

# print("\n")
# print(model.predict(embeddings[-1].reshape(1, -1)))

########### Plotting ####################

#plot_data(X)
#model2.plot_clusters(X, model2.labels, model2.centroids)


########3 Plot example ##################
# n = ['Ariel_Sharon', 'Arnold_Schwarzenegger','Colin_Powell']

# for i in subset_labels:
#   color = colors[i]
#   for feature in subset_embeddings[subset_labels == i]:
#       plt.scatter(feature[0], feature[1], marker="o", color=color, s=20, label=color)

# leg = plt.legend(n, loc="lower right")
# c=["g","r","c"]

# for i, j in enumerate(leg.legendHandles):
#     j.set_color(c[i])

# plt.show()

##for unknown in unknowns:
##    classification = model.predict(unknown)
##    plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)
