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
from evaluate import Evaluate
import matplotlib.cm as cm


embeddings = np.round(np.load("embeddings/test/embeddings.npy"), decimals=6)
t_labels = np.load("embeddings/test/labels.npy")
label_strings = np.load("embeddings/test/label_strings.npy")

# 3 classes and 15 images
# subset_embeddings = embeddings[:15]
# subset_labels = t_labels[:15]

# Clustering
X = np.round(embeddings[:-1], decimals=6)
#X = subset_embeddings
# label: names
names = label_strings[:-1]

# KMeans from sklearn
kmeans = KMeans(n_clusters=10).fit(X)
print(kmeans.labels_)
# print("\n")
# print(kmeans.cluster_centers_)
# print("\n")
# y = embeddings[-1].reshape(1, -1)
# p = kmeans.predict(y)
# print(p)

# KMeans implementation
model1 = K_Means(num_clusters=10)
model1.fit(X)

model2 = K_Means(num_clusters=10, k_means_plus_plus=True)
model2.fit(X)

#print(model.centroids)
# print("\n")
print(model1.labels)

print(model2.labels)
# print("\n")
# print(model.label_feature)

########### Prediction ###################

# print("\n")
# print(model.predict(embeddings[-1].reshape(1, -1)))
