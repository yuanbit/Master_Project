import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


class K_Means:
    
    def __init__(self, num_clusters, max_iter=300, k_means_plus_plus=False):

        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.k_means_plus_plus = k_means_plus_plus
        

    def initialize_centroids(self, data):

        # Dimension: k * number of features
        centroids = data[np.random.choice(data.shape[0], self.num_clusters, replace=False)]

        return centroids

    def kmeans_plus_plus(self, data):

        # Get random index from data
        i = np.random.randint(0, data.shape[0])

        # Initialize first random centroid
        centroid = np.array([data[i]])

        # Substract the dataset with the first chosen centroid
        new_data = np.delete(data, (i), axis=0)

        # Pick the next k-1 centroids
        for k in range(1, self.num_clusters):

            # Get the distance of the data points to the nearest centroid
            D = np.array([])

            # For each datapoint
            for x in new_data:
                # Compute the squared Euclidian distance from the nearest centroid (with minimum distance)
                D = np.append(D, np.min(np.square(np.linalg.norm(x - centroid))))

            # Probability of choosing the next centroid (proportional to ||c_i-x||^2)
            # These are the probabilities of choosing the particular data point as the next centroid
            prob= D/np.sum(D)
            cummulative_prob= np.cumsum(prob)

            r = np.random.random()
            # Get index of the first cum prob that is greater than chosen random number
            #idx = np.where(cummulative_prob >= r)[0][0]

            idx = np.argmax(prob)

            # Append the next centroid
            centroid = np.append(centroid,[new_data[idx]],axis=0)

            # Substract the dataset with the chosen centroids
            new_data = np.delete(new_data, (idx), axis=0)
            
        return centroid


    def distance(self, data, centroids):

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
            centroids[i, :] = np.mean(data[labels == i, :], axis=0)

        return centroids

    def fit(self, data):

        if self.k_means_plus_plus == False:
            self.centroids = self.initialize_centroids(data)
        else:
            self.centroids = self.kmeans_plus_plus(data)

        for i in range(self.max_iter):

            prev_centroids = self.centroids

            distances = self.distance(data, prev_centroids)
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

        distance = self.distance(data, self.centroids)

        return self.compute_closest_centroid(distance)

    def get_cluster_labels(self, labels, names):

        # dictionary with cluster id and array of names of faces
        label_name = {}

        for i in range(self.num_clusters): 
            label_name[i] = []

        for j in range(len(labels)):
            label_name[labels[j]].append(names[j])

        # Sort results in alphabetical order
        label_name_sorted = sorted(label_name.items() ,  key=lambda x: x[1])

        clusters = []

        for i in range(len(label_name_sorted)):
            clusters.append(label_name_sorted[i][1])

        return clusters


##### Testing #################

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [1,3],
              [8,9],
              [0,3],
              [5,4],
              [6,4],])




# Y = np.array([[1,1], [8, 8.5]])

# model = K_Means(num_clusters=3)
# model.fit(X)

# # print(model.centroids)
# # print("\n")
# print(model.labels)
# print("\n")
# print(model.label_feature)

# print("\n")
# print(model.predict(Y))


########### Plotting ####################
# #plt.scatter(X[:,0], X[:,1], s=150)

# k = 3
# #colors of plot
# colors = 10*["g","r","c","b","k"]

# #plot centroids
# for k in range(k):

#     plt.scatter(model.centroids[k][0], model.centroids[k][1], marker="o", color="k", s=60, linewidths=5)

# for i in model.labels:
#   color = colors[i]
#   for feature in X[model.labels == i]:
#       plt.scatter(feature[0], feature[1], marker="x", color=color, s=50, linewidths=5)

# plt.show()



