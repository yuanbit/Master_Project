import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import sys
import numpy as np
from sklearn.cluster import KMeans
from k_means import K_Means

## TODO: create python dict: keys: label_strings, values: embeddings

embeddings = np.round(np.load("embeddings/embeddings.npy"), decimals=6)
labels = np.load("embeddings/labels.npy")
label_strings = np.load("embeddings/label_strings.npy")

embedding_label = {}

for i in range(len(embeddings)-1):
	embedding_label[label_strings[i]] = embeddings[i] 

#print(embedding_label)

# Clustering
X = embeddings[:-1]

# KMeans from sklearn
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
l = kmeans.labels_
print(l)
# y = embeddings[-1].reshape(1, -1)
# p = kmeans.predict(y)
# print(p)

# KMeans implementation
model = K_Means(k=10)
model.fit(X)

# colors of plot
colors = 10*["g","r","c","b","k"]

# plot centroids
for centroid in model.centroids:
    plt.scatter(model.centroids[centroid][0], model.centroids[centroid][1],
                marker="o", color="k", s=50, linewidths=5)

# plot datapoints in each colored cluster
for classification in model.classes:
    color = colors[classification]
    for feature in model.classes[classification]:
        plt.scatter(feature[0], feature[1], marker="x", color=color, s=40, linewidths=5)

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

# Datapoints in each cluster
#print(model.classes)

clusters = model.classes

#for i in clusters.get(0):
	#print(i)

a = np.round(clusters.get(0)[0], decimals=6)

# print(a)
# print(embeddings[0])

# print(np.array_equal(a, embeddings[0]))

# fc = []

# for i in clusters.get(1):
# 	for j in range(len(embeddings)):
# 		if np.array_equal(i, embeddings[j]) == True:
# 			print(label_strings[j])

print(len(clusters))

for key, value in clusters.items():
	print(key)
	#print(len(value))
