import numpy as np
import math
from numpy import random

from sklearn.metrics import confusion_matrix

true = random.random_integers(0, 1, 10000)
sigma_0 = 1
sigma_1 = 4
mu = 0

border = 8*math.sqrt(math.log(2)) / math.sqrt(15)

predicted = []

for y in true:
    if y == 0:
        x_1, x_2 = np.random.normal(mu, sigma_0, 2)
    else:
        x_1, x_2 = np.random.normal(mu, sigma_1, 2)

    if math.sqrt(x_1**2 + x_2**2) <= border:
        predicted.append(0)
    else:
        predicted.append(1)


matrix = confusion_matrix(true, predicted)

print float(matrix[0][1] + matrix[1][0])/10000
