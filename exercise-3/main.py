import heapq
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats.mstats import mode
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

sigma_0 = 1
sigma_1 = 4
mu = 0
TEST_SIZE = 5000
TRAINING_SIZE = 20000

""" PART B """

# for y = 1
training_y_1_x_1 = np.random.normal(mu, sigma_1, TRAINING_SIZE)
training_y_1_x_2 = np.random.normal(mu, sigma_1, TRAINING_SIZE)

plt.plot(training_y_1_x_1, training_y_1_x_2, 'bo')

# for y = 0
training_y_0_x_1 = np.random.normal(mu, sigma_0, TRAINING_SIZE)
training_y_0_x_2 = np.random.normal(mu, sigma_0, TRAINING_SIZE)

plt.plot(training_y_0_x_1, training_y_0_x_2, 'ro')
plt.axis([-15, 15, -15, 15])

plt.show()
plt.close() # clear previous figure

""" PART C """

# for y = 1
test_y_1_x_1 = np.random.normal(mu, sigma_1, TEST_SIZE)
test_y_1_x_2 = np.random.normal(mu, sigma_1, TEST_SIZE)

plt.plot(test_y_1_x_1, test_y_1_x_2, 'bo')

# for y = 0
test_y_0_x_1 = np.random.normal(mu, sigma_0, TEST_SIZE)
test_y_0_x_2 = np.random.normal(mu, sigma_0, TEST_SIZE)

plt.plot(test_y_0_x_1, test_y_0_x_2, 'ro')
plt.axis([-15, 15, -15, 15])

plt.show()
plt.close() # clear previous figure

""" PART D """

training_y_0 = [(training_y_0_x_1[i], training_y_0_x_2[i]) for i in range(TRAINING_SIZE)]
training_y_1 = [(training_y_1_x_1[i], training_y_1_x_2[i]) for i in range(TRAINING_SIZE)]
training_set = training_y_0 + training_y_1 # first 500 are y=0, last 500 are y=1

test_y_0 = [(test_y_0_x_1[i], test_y_0_x_2[i]) for i in range(TEST_SIZE)]
test_y_1 = [(test_y_1_x_1[i], test_y_1_x_2[i]) for i in range(TEST_SIZE)]
test_set = test_y_0 + test_y_1 # first 2000 are y=0, last 2000 are y=1

true_labels = [0] * TEST_SIZE + [1] * TEST_SIZE

ks = [1, 3, 5, 7, 9, 13, 17, 21, 25, 33, 41, 49, 57]

predicted = []
for k in ks:
    nbrs = NearestNeighbors(n_neighbors=k).fit(training_set)
    distances, indices = nbrs.kneighbors(test_set)
    print indices
    for closest in indices:
        labels = [0 if i < TRAINING_SIZE else 1 for i in closest]
        label, occurrence = mode(labels)
        predicted.append(int(label))

matrix = confusion_matrix(true_labels, predicted)
print matrix
