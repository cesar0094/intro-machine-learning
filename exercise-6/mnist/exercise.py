import numpy as np
from scipy.spatial import distance

import mnist_load_show

SAMPLE_SIZE = 500
student_ID = '014632888'


def k_means(data, means):

    change = True
    new_means = np.zeros(means.shape)
    while change:
        # (SAMPLE_SIZE, 10)
        distances = distance.cdist(data, means, 'euclidean')
        cluster_indices = [ np.argmin(d) for d in distances ]
        new_means = np.zeros(means.shape)
        num = [0.0 for i in range(len(means))]
        for i, c in enumerate(cluster_indices):
            new_means[c] += data[i]
            num[c] += 1

        new_means = [ new_means[i] / n for i, n in enumerate(num)]

        change = False

        for i in range(len(new_means)):
            if (new_means[i] != means[i]).all():
                change = True
                break
        means = np.copy(new_means)

    distances = distance.cdist(data, means, 'euclidean')
    cluster_indices = np.array([ np.argmin(d) for d in distances ])
    clusters = np.array([ data[cluster_indices == i] for i in range(len(means)) ])
    return np.array(new_means), clusters

def k_mediods(data, mediods):

    change = True
    new_mediods = np.zeros(mediods.shape)

    while change:
        # (data.size, mediods.size)
        distances = distance.cdist(data, mediods, 'euclidean')
        cluster_indices = np.array([ np.argmin(d) for d in distances ])
        new_mediods = np.zeros(mediods.shape)
        for i in range(len(mediods)):
            curr_cluster_indices = np.argwhere(cluster_indices == i).flatten()
            cluster = data[curr_cluster_indices]
            distances = distance.cdist(cluster, cluster , 'euclidean')
            new_mean_i = np.argmin(np.sum(distances, axis=1))
            new_mediods[i] = cluster[new_mean_i]

        change = False

        for i in range(len(new_mediods)):
            if (new_mediods[i] != mediods[i]).all():
                change = True
                break
        mediods = np.copy(new_mediods)

    clusters = np.array([ data[cluster_indices == i] for i in range(len(mediods)) ])
    return new_mediods, clusters


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

cluster_mediods, clusters = k_mediods(X, first_ten)
mnist_load_show.visualize(cluster_mediods)
for mediod, cluster in zip(cluster_mediods, clusters):
    mnist_load_show.visualize(np.insert(cluster, 0, mediod, axis=0))

cluster_mediods, clusters = k_mediods(X, first_label_instance)
mnist_load_show.visualize(cluster_mediods)
for mediod, cluster in zip(cluster_mediods, clusters):
    mnist_load_show.visualize(np.insert(cluster, 0, mediod, axis=0))
