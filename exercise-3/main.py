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
mu_0 = 0
mu_1 = 0
TEST_SIZE = 2000
TRAINING_SIZE = 500

""" PART B """

rand_y = random.random_integers(0, 1, TRAINING_SIZE)

training_y_0_x_1 = []
training_y_0_x_2 = []
training_y_1_x_1 = []
training_y_1_x_2 = []
for y in rand_y:
    if y == 0:
        training_y_0_x_1.append(np.random.normal(mu_0, sigma_0, 1)[0])
        training_y_0_x_2.append(np.random.normal(mu_0, sigma_0, 1)[0])
    else:
        training_y_1_x_1.append(np.random.normal(mu_1, sigma_1, 1)[0])
        training_y_1_x_2.append(np.random.normal(mu_1, sigma_1, 1)[0])

plt.plot(training_y_1_x_1, training_y_1_x_2, 'bo')

plt.plot(training_y_0_x_1, training_y_0_x_2, 'ro')
plt.axis([-15, 15, -15, 15])

plt.show()
plt.close() # clear previous figure

""" PART C """

rand_y = random.random_integers(0, 1, TEST_SIZE)

test_y_0_x_1 = []
test_y_0_x_2 = []
test_y_1_x_1 = []
test_y_1_x_2 = []
for y in rand_y:
    if y == 0:
        test_y_0_x_1.append(np.random.normal(mu_0, sigma_0, 1)[0])
        test_y_0_x_2.append(np.random.normal(mu_0, sigma_0, 1)[0])
    else:
        test_y_1_x_1.append(np.random.normal(mu_1, sigma_1, 1)[0])
        test_y_1_x_2.append(np.random.normal(mu_1, sigma_1, 1)[0])

plt.plot(test_y_1_x_1, test_y_1_x_2, 'bo')

plt.plot(test_y_0_x_1, test_y_0_x_2, 'ro')
plt.axis([-15, 15, -15, 15])

plt.show()
plt.close() # clear previous figure

""" PART D """

training_y_0 = [(training_y_0_x_1[i], training_y_0_x_2[i]) for i in range(len(training_y_0_x_1))]
training_y_1 = [(training_y_1_x_1[i], training_y_1_x_2[i]) for i in range(len(training_y_1_x_1))]
training_set = training_y_0 + training_y_1
true_training_labels = [0] * len(training_y_0) + [1] * len(training_y_1)

test_y_0 = [(test_y_0_x_1[i], test_y_0_x_2[i]) for i in range(len(test_y_0_x_1))]
test_y_1 = [(test_y_1_x_1[i], test_y_1_x_2[i]) for i in range(len(test_y_1_x_1))]
test_set = test_y_0 + test_y_1
true_test_labels = [0] * len(test_y_0) + [1] * len(test_y_1)

ks = [1, 3, 5, 7, 9, 13, 17, 21, 25, 33, 41, 49, 57]

errors = []
for k in ks:
    predicted = []
    nbrs = NearestNeighbors(n_neighbors=k).fit(training_set)
    distances, indices = nbrs.kneighbors(test_set)
    for closest in indices:

        labels = [0 if i < len(training_y_0) else 1 for i in closest]
        label, occurrence = mode(labels)
        predicted.append(int(label))

    matrix = confusion_matrix(true_test_labels, predicted)
    errors.append(float(matrix[0][1] + matrix[1][0]) / len(true_test_labels))

test_error, = plt.plot(ks, errors, 'ro-', label='Test')

errors = []
for k in ks:
    predicted = []
    nbrs = NearestNeighbors(n_neighbors=k).fit(training_set)
    distances, indices = nbrs.kneighbors(training_set)
    for closest in indices:

        labels = [0 if i < len(training_y_0) else 1 for i in closest]
        label, occurrence = mode(labels)
        predicted.append(int(label))

    matrix = confusion_matrix(true_training_labels, predicted)
    errors.append(float(matrix[0][1] + matrix[1][0]) / len(true_training_labels))

bayes_error = [.155] * len(ks)
plt.plot(ks, bayes_error, 'g-', label="Bayes")
plt.plot(ks, errors, 'bo-', label="Train")

plt.xlabel('Number of Nearest Neighbors')
plt.ylabel('Error Rate')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

plt.show()
plt.close() # clear previous figure
