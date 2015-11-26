import math
import numpy
from numpy import random
from scipy import optimize
import matplotlib.pyplot as plt

NUM_SAMPLES = 30
FOLD_SIZE = 10
K = 10

def fit_func(x, *args):
    return sum(a * x ** i for i, a in enumerate(args))

x = random.uniform(-3, 3, NUM_SAMPLES)
f_x = lambda x: 2 + x - 0.5 * x**2

mean = 0
sigma = 0.4
error = random.normal(mean, sigma, NUM_SAMPLES)
y = [ f_x(x[i]) + error[i] for i in range(NUM_SAMPLES) ]
y_mean = numpy.mean(y)

x_aprox = numpy.arange(-3, 3, 0.2)

for k in range(1, K+2):

    x0 = numpy.ones(k)
    values, cov_matrix = optimize.curve_fit(fit_func, x, y, x0)
    y_aprox = [fit_func(i, *values) for i in x_aprox]

    top = sum([ (y[i] - fit_func(x_i, *values))**2 for i, x_i in enumerate(x) ])
    down = sum([ (y[i] - y_mean)**2 for i, x_i in enumerate(x) ])

    r_sqrd = 1 - top/down
    print "R^2:", r_sqrd

    plt.plot(x, y, 'bo')
    plt.plot(x_aprox, y_aprox, 'gx')
    plt.show()
    plt.close()

n_partitions = int(math.ceil(NUM_SAMPLES/float(FOLD_SIZE)))

x_partitions = [ x[FOLD_SIZE*i:FOLD_SIZE*(i+1)] for i in range(n_partitions) ]
y_partitions = [ y[FOLD_SIZE*i:FOLD_SIZE*(i+1)] for i in range(n_partitions) ]

error_sums = []
for k in range(1, K+2):
    error_sum = 0
    for i in range(n_partitions):
        training_x = numpy.array([x_partitions[j] for j in range(n_partitions) if j != i ]).flatten()
        training_y = numpy.array([y_partitions[j] for j in range(n_partitions) if j != i ]).flatten()

        x0 = numpy.zeros(k)
        values, cov_matrix = optimize.curve_fit(fit_func, training_x, training_y, x0)
        y_aprox = [fit_func(j, *values) for j in x_partitions[i]]

        sqr_err_sum = sum([ (training_y[i] - fit_func(x_i, *values))**2 for i, x_i in enumerate(training_x) ])

        error_sum += sqr_err_sum

    error_sums.append(error_sum)

plt.plot(error_sums, color='blue', lw=2)
plt.yscale('log')
plt.show()
