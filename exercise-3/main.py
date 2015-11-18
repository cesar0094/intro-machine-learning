import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

sigma_0 = 1
sigma_1 = 4
mu = 0

""" PART B """

# for y = 1
training_y_1_x_1 = np.random.normal(mu, sigma_1, 500)
training_y_1_x_2 = np.random.normal(mu, sigma_1, 500)

plt.plot(training_y_1_x_1, training_y_1_x_2, 'bo')

# for y = 0
training_y_0_x_1 = np.random.normal(mu, sigma_0, 500)
training_y_0_x_2 = np.random.normal(mu, sigma_0, 500)

plt.plot(training_y_0_x_1, training_y_0_x_2, 'ro')
plt.axis([-15, 15, -15, 15])

plt.show()
plt.close() # clear previous figure

""" PART C """

# for y = 1
test_y_1_x_1 = np.random.normal(mu, sigma_1, 2000)
test_y_1_x_2 = np.random.normal(mu, sigma_1, 2000)

plt.plot(test_y_1_x_1, test_y_1_x_2, 'bo')

# for y = 0
test_y_0_x_1 = np.random.normal(mu, sigma_0, 2000)
test_y_0_x_2 = np.random.normal(mu, sigma_0, 2000)

plt.plot(test_y_0_x_1, test_y_0_x_2, 'ro')
plt.axis([-15, 15, -15, 15])

plt.show()
plt.close() # clear previous figure

""" PART D """

training_y_0 = [(training_y_0_x_1[i], training_y_0_x_2[i]) for i in range(500)]
training_y_1 = [(training_y_1_x_1[i], training_y_1_x_2[i]) for i in range(500)]
training_set = training_y_0 + training_y_1 # first 500 are y=0, last 500 are y=1

test_y_0 = [(test_y_0_x_1[i], test_y_0_x_2[i]) for i in range(2000)]
test_y_1 = [(test_y_1_x_1[i], test_y_1_x_2[i]) for i in range(2000)]
test_set = test_y_0 + test_y_1 # first 2000 are y=0, last 2000 are y=1

k = [1, 3, 5, 7, 9, 13, 17, 21, 25, 33, 41, 49, 57]

distances = cdist(test_set, training_set, 'euclidean')
