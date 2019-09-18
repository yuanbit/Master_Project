import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np


class K_Means:

    def __init__(self, k=2, tol=0.0001, max_iter=300):
        # tol -- how much will the centroid move (percentage)
        # max_iter -- how many times to run iteration
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}

        # Initialize centroids to the first k datapoints

        subset = data[np.random.choice(data.shape[0], self.k, replace=False)]

        for i in range(self.k):
            #idx = np.random.randint(len(data))
            self.centroids[i] = subset[i]
            #self.centroids[i] = data[i]

        for i in range(self.max_iter):
            # Cluster assignments dictionary
            # Keys: centroids
            # Values: datapoints
            self.classes = {}

            for i in range(self.k):
                self.classes[i] = []

            # Iteratre through the features of each datapoint and the features of the centroid
            for feature in data:
                # Compute Euclidean distance between datapoint and the centroids
                distances = [np.linalg.norm(feature-self.centroids[centroid]) for centroid in self.centroids]
                # Assign datapoint to the cluster with the least distance
                classification = distances.index(min(distances))
                # Add feature vector of datapoint to cluster assignments dictionary
                self.classes[classification].append(feature)

            # Centroids of the previous iteration
            prev_centroids = dict(self.centroids)

            # Iterate through the clusters
            for classification in self.classes:
                # Compute the average of the datapoints assigned to each centroid
                self.centroids[classification] = np.average(self.classes[classification],axis=0)

            # Check optimal values of centroids
            optimized = True

            # Iterate through each new centroid
            for centroid in self.centroids:

                original_centroid = prev_centroids[centroid]
                current_centroid = self.centroids[centroid]
                # Compare current centroids with the centroids of the previous iteration
                # If the percentage change in the centroids is greater than the tolerance
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    # show percentage change in each iterations
                    #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False
            # Centroids don't change more than tolerance
            if optimized:
                break

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

# X = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8 ],
#               [8, 8],
#               [1, 0.6],
#               [9,11],
#               [1,3],
#               [8,9],
#               [0,3],
#               [5,4],
#               [6,4],])



#plt.scatter(X[:,0], X[:,1], s=150)
#plt.show()

#colors of plot
# colors = 10*["g","r","c","b","k"]

# clf = K_Means(k=3)
# clf.fit(X)


# # plot centroids
# for centroid in clf.centroids:
#     plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
#                 marker="o", color="k", s=70, linewidths=5)

# # plot datapoints in each colored cluster
# for classification in clf.classes:
#     color = colors[classification]
#     for feature in clf.classes[classification]:
#         plt.scatter(feature[0], feature[1], marker="x", color=color, s=50, linewidths=5)

# plt.show()
# # # # Predict new datapoint
# # # ##unknowns = np.array([[1,3],
# # # ##                     [8,9],
# # # ##                     [0,3],
# # # ##                     [5,4],
# # # ##                     [6,4],])
# # # ##
# # # ##for unknown in unknowns:
# # # ##    classification = clf.predict(unknown)
# # # ##    plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)
# # # ##

# plt.show()

# # # TODO: print the datapoints in each cluster?
# # print(clf.classes)