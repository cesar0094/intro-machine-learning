import numpy as np
from scipy.spatial import distance

import mnist_load_show

SAMPLE_SIZE = 500
student_ID = '014632888'


def k_means(data, means):

    changed = True
    while changed:
        distances = distance.cdist(data, means, 'sqeuclidean')
        cluster_indices = np.argmin(distances, axis=1)
        new_means = np.array([ data[cluster_indices == i].mean(axis=0) for i in range(len(means)) ])
        changed = not (new_means == means).all()
        means = np.copy(new_means)

    clusters = np.array([ data[cluster_indices == i] for i in range(len(means)) ])
    return np.array(new_means), clusters

def k_medoids(dissimilarity, mediod_indices):

    changed = True
    mediod_indices = np.array(mediod_indices)
    while changed:
        dissimilarity_medoids = np.array([ dissimilarity[mediod_index] for mediod_index in mediod_indices]).transpose()
        cluster_indices = np.argmin(dissimilarity_medoids, axis=1)
        new_mediod_indices = np.zeros(mediod_indices.shape)

        for i in range(len(mediod_indices)):
            curr_cluster_indices = np.argwhere(cluster_indices == i).flatten()
            reduced_dissimilarity = dissimilarity[curr_cluster_indices].transpose()[curr_cluster_indices]
            new_mediod_reduced_i = np.argmin(np.sum(reduced_dissimilarity, axis=1))
            new_mediod_i = curr_cluster_indices[new_mediod_reduced_i]
            new_mediod_indices[i] = new_mediod_i

        changed = (new_mediod_indices != mediod_indices).any()
        mediod_indices = np.copy(new_mediod_indices)

    return new_mediod_indices, cluster_indices


X, Y = mnist_load_show.read_mnist_training_data(SAMPLE_SIZE)

first_ten = X[:10]
# select first instance of each label
first_label_instance = np.array([ X[np.where(Y == i)[0][0]] for i in range(10) ])

cluster_means, clusters = k_means(X, first_ten)
mnist_load_show.visualize(cluster_means)
for mean, cluster in zip(cluster_means, clusters):
    mnist_load_show.visualize(np.insert(cluster, 0, mean, axis=0))

cluster_means, clusters = k_means(X, first_label_instance)
mnist_load_show.visualize(cluster_means)
for mean, cluster in zip(cluster_means, clusters):
    mnist_load_show.visualize(np.insert(cluster, 0, mean, axis=0))

distances = distance.cdist(X, X, 'euclidean')
cluster_medoids_indices, clusters_indices = k_medoids(distances, list(range(10)))
cluster_medoids = np.array([X[int(i)] for i in cluster_medoids_indices])
clusters = np.array([ X[clusters_indices == i] for i in range(10) ])
mnist_load_show.visualize(cluster_medoids)
for mediod, cluster in zip(cluster_medoids, clusters):
    mnist_load_show.visualize(np.insert(cluster, 0, mediod, axis=0))

mediod_indices = [ np.where(Y == i)[0][0] for i in range(10) ]
cluster_medoids_indices, clusters_indices = k_medoids(distances, mediod_indices)
cluster_medoids = np.array([X[int(i)] for i in cluster_medoids_indices])
clusters = np.array([ X[clusters_indices == i] for i in range(10) ])
mnist_load_show.visualize(cluster_medoids)
for mediod, cluster in zip(cluster_medoids, clusters):
    mnist_load_show.visualize(np.insert(cluster, 0, mediod, axis=0))
