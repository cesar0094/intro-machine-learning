import numpy as np
from scipy.spatial import distance

import mnist_load_show

SAMPLE_SIZE = 500
student_ID = '014632888'


def k_means(data, means):

    change = True

    while change:
        # (SAMPLE_SIZE, 10)
        distances = distance.cdist(data, means, 'euclidean')
        clusters = [ np.argmin(d) for d in distances ]
        new_means = np.zeros(means.shape)
        num = [0.0 for i in range(len(means))]
        for i, c in enumerate(clusters):
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

X, Y = mnist_load_show.read_mnist_training_data(SAMPLE_SIZE)

cluster_means = X[:10]
cluster_means = k_means(X, cluster_means)

mnist_load_show.visualize(cluster_means)

# select first instance of each label
cluster_means = [X[np.where(Y == i)[0][0]] for i in range(10)]
cluster_means = np.array(cluster_means)
mnist_load_show.visualize(cluster_means)

cluster_means = k_means(X, cluster_means)
mnist_load_show.visualize(cluster_means)
