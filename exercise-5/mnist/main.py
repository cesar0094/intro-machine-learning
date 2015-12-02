import numpy
from numpy import random
from sklearn.metrics import confusion_matrix

import mnist_load_show

TRAINING_SIZE = 30000
TEST_SIZE = 30000

X, Y = mnist_load_show.read_mnist_training_data()

training_set = X[:TRAINING_SIZE]
training_labels = Y[:TRAINING_SIZE]
test_set = X[TRAINING_SIZE : TRAINING_SIZE + TEST_SIZE]
test_labels = Y[TRAINING_SIZE: TRAINING_SIZE + TEST_SIZE]


# adding the bias column
training_set = numpy.insert(training_set, 0, 1, axis=1)
test_set = numpy.insert(test_set, 0, 1, axis=1)

# learning rate
rate = 0.5

def predict_label(x, W):
    x = numpy.matrix(x).transpose()
    return numpy.argmax(W.dot(x))

def calculate_label(x, w):
    return numpy.sign(numpy.dot(x, w))

""" One vs. All """

W = [ numpy.zeros(dimensions) for i in range(10) ]
for label in range(10):

    score = 0
    score_change = True
    top_score = -1
    top_w = 0

    while score_change:
        score_change = False

        for i, x in enumerate(training_set):
            y_i = 1 if training_labels[i] == label else -1
            y_hat_i = calculate_label(W[label], x)

            if y_i != y_hat_i:

                # let's check if it was a good performing W
                if score > top_score:
                    score_change = True
                    top_score = score
                    top_w = numpy.copy(W[label])

                score = 0
                W[label] = W[label] + y_i * x

            else:
                score += 1

    W[label] = top_w

# since we modify W[i] with top_w several times, we avoid making it a matrix until
# it is finalized
W = numpy.matrix(W)

predicted_labels = [predict_label(x, W) for x in test_set]

matrix = confusion_matrix(test_labels, predicted_labels)
print "One vs. All"
print matrix
