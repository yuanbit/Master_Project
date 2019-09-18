import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


class K_Means:
    '''Implementing Kmeans algorithm.'''

    def __init__(self, num_clusters, max_iter=300):

        self.num_clusters = num_clusters
        self.max_iter = max_iter
        

    def initialize_centroids(self, data):

        # Dimension: k * number of features
        centroids = data[np.random.choice(data.shape[0], self.num_clusters, replace=False)]

        return centroids


    def distance_from_centroid(self, data, centroids):

        # Initialize matrix for the distances of each data point to the centroids
        # Dimension: number of data points * k
        # Rows: data point
        # Cols: centroid
        distances = np.zeros((data.shape[0], self.num_clusters))

        # For each centroid
        for i in range(self.num_clusters):
            # Compute Euclidean distance to data point
            row_norm = np.linalg.norm(data - centroids[i, :], axis=1)
            # Set distance in matrix
            distances[:, i] = row_norm

        return distances

    def compute_closest_centroid(self, distances):

        # Return the centroid with the minimum distance
        return np.argmin(distances, axis=1)


    def compute_new_centroids(self, data, labels):

        # Initialize dimension for new centroids
        centroids = np.zeros((self.num_clusters, data.shape[1]))

        for i in range(self.num_clusters):
            # Compute the average of all data points for each label (cluster)
            # Assign new centroids for each cluster as the average 
            if len(data[labels == 0, :]) != 0:
                centroids[i, :] = np.mean(data[labels == i, :], axis=0)

        return centroids

    def fit(self, data):

        self.centroids = self.initialize_centroids(data)

        for i in range(self.max_iter):

            prev_centroids = self.centroids

            distances = self.distance_from_centroid(data, prev_centroids)
            # The indices represent the row indices of the data
            self.labels = self.compute_closest_centroid(distances)
            # Compute new centroids
            # Current centroid
            self.centroids = self.compute_new_centroids(data, self.labels)

            # Terminate when the current and previous centroids are equal
            if np.all(prev_centroids == self.centroids):
                break

        # Create dictionary
        self.label_feature = {}

        for i in range(self.num_clusters):
            
            self.label_feature[i] = []

            for feature in data[self.labels == i]:

                self.label_feature[i].append(feature)

    def predict(self, data):

        distance = self.distance_from_centroid(data, self.centroids)

        return self.compute_closest_centroid(distance)


##### Testing #################

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

# Y = np.array([[1,1], [8, 8.5]])

# model = K_Means(num_clusters=3)
# model.fit(X)

# print(model.centroids)
# print("\n")
# print(model.labels)
# print("\n")
# print(model.label_feature)

# print("\n")
# print(model.predict(Y))


########### Plotting ####################
# #plt.scatter(X[:,0], X[:,1], s=150)

# #colors of plot
# colors = 10*["g","r","c","b","k"]

# #plot centroids
# for k in range(k):

#     plt.scatter(centroids[k][0], centroids[k][1], marker="o", color="k", s=60, linewidths=5)

# for i in labels:
#   color = colors[i]
#   for feature in X[labels == i]:
#       plt.scatter(feature[0], feature[1], marker="x", color=color, s=50, linewidths=5)

# plt.show()



