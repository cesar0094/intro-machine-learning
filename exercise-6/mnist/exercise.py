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

    return np.array(new_means)

def k_mediods(data, means):

    change = True
    new_means = np.zeros(means.shape)

    while change:
        # (data.size, means.size)
        distances = distance.cdist(data, means, 'euclidean')
        cluster_indices = np.array([ np.argmin(d) for d in distances ])
        new_means = np.zeros(means.shape)
        for i in range(len(means)):
            curr_cluster_indices = np.argwhere(cluster_indices == i).flatten()
            cluster = data[curr_cluster_indices]
            distances = distance.cdist(cluster, cluster , 'euclidean')
            new_mean_i = np.argmin(np.sum(distances, axis=1))
            new_means[i] = cluster[new_mean_i]

        change = False

        for i in range(len(new_means)):
            if (new_means[i] != means[i]).all():
                change = True
                break
        means = np.copy(new_means)

    return np.array(new_means)


X, Y = mnist_load_show.read_mnist_training_data(SAMPLE_SIZE)

first_ten = X[:10]
cluster_means = k_means(X, first_ten)
mnist_load_show.visualize(cluster_means)

# select first instance of each label
first_label_instance = np.array([ X[np.where(Y == i)[0][0]] for i in range(10) ])
cluster_means = k_means(X, first_label_instance)
mnist_load_show.visualize(cluster_means)

cluster_means = k_mediods(X, first_ten)
mnist_load_show.visualize(cluster_means)

cluster_means = k_mediods(X, first_label_instance)
mnist_load_show.visualize(cluster_means)
