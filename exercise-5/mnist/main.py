import numpy
from numpy import random
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import mnist_load_show

TRAINING_SIZE = 30000
TEST_SIZE = 30000
NUM_EPOCS = 10000

X, Y = mnist_load_show.read_mnist_training_data()

training_set = X[:TRAINING_SIZE]
training_labels = Y[:TRAINING_SIZE]
test_set = X[TRAINING_SIZE : TRAINING_SIZE + TEST_SIZE]
test_labels = Y[TRAINING_SIZE: TRAINING_SIZE + TEST_SIZE]


# adding the bias column
training_set = numpy.insert(training_set, 0, 1, axis=1)
test_set = numpy.insert(test_set, 0, 1, axis=1)
dimensions = len(training_set[0])

# learning rate
rate = 0.5

def predict_one_vs_all_label(x, W):
    x = x.transpose()
    return numpy.argmax(W.dot(x))

def predict_all_vs_all_label(x, W):
    x = x.transpose()
    sums_j = [numpy.sum(numpy.dot(w_i, x)) for w_i in W]
    return numpy.argmax(sums_j)

def calculate_label(x, w):
    return numpy.sign(numpy.dot(x, w))

def get_learning_weights(training_set, training_labels, label):

    dimensions = len(training_set[0])
    weights = numpy.zeros(dimensions)
    score = 0
    score_change = True
    top_score = -1
    top_w = 0
    epocs = 0

    while score_change and epocs < NUM_EPOCS:
        score_change = False

        for i, x in enumerate(training_set):
            y_i = 1 if training_labels[i] == label else -1
            y_hat_i = calculate_label(weights, x)

            if y_i != y_hat_i:

                # let's check if it was a good performing W
                if score > top_score:
                    score_change = True
                    top_score = score
                    top_w = numpy.copy(weights)

                score = 0
                weights = weights + y_i * x

            else:
                score += 1

        epocs += 1

    return top_w

""" One vs. All """

W = [ get_learning_weights(training_set, training_labels, label) for label in range(10) ]
W = numpy.matrix(W)

predicted_labels = [predict_one_vs_all_label(x, W) for x in test_set]

matrix = confusion_matrix(test_labels, predicted_labels)
print "One vs. All"
print matrix
print metrics.classification_report(test_labels, predicted_labels)

""" All vs. All """
W = [ numpy.zeros((10,dimensions)) for i in range(10) ]

for i_label in range(10):
    for j_label in range(10):
        if i_label == j_label:
            continue

        # only consider two labels
        boolean_array = numpy.logical_or(training_labels == i_label, training_labels == j_label)
        reduced_training_labels = training_labels[boolean_array]
        reduced_training_set = training_set[boolean_array]

        W[i_label][j_label] = get_learning_weights(reduced_training_set, reduced_training_labels, i_label)

predicted_labels = [predict_all_vs_all_label(x, W) for x in test_set]

matrix = confusion_matrix(test_labels, predicted_labels)
print "All vs. All"
print matrix
print metrics.classification_report(test_labels, predicted_labels)
