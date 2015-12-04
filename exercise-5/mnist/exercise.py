import numpy as np
import mnist_load_show
from sklearn.metrics import confusion_matrix
"""
============================================
DO NOT FORGET TO INCLUDE YOUR STUDENT ID
============================================
"""
student_ID = '014632888'

TRAINING_SIZE = 30000
TEST_SIZE = 30000
NUM_EPOCHS = 10

X, Y = mnist_load_show.read_mnist_training_data()

training_set = X[:TRAINING_SIZE]
training_labels = Y[:TRAINING_SIZE]
test_set = X[TRAINING_SIZE : TRAINING_SIZE + TEST_SIZE]
test_labels = Y[TRAINING_SIZE: TRAINING_SIZE + TEST_SIZE]


# adding the bias column
training_set = np.insert(training_set, 0, 1, axis=1)
test_set = np.insert(test_set, 0, 1, axis=1)
DIMENSIONS = len(training_set[0])

def predict_one_vs_all_label(x, W):
    x = x.transpose()
    return np.argmax(W.dot(x))

def predict_all_vs_all_label(x, W):
    x = x.transpose()
    sums_j = [np.sum(np.sign(np.dot(w_i, x))) for w_i in W]
    return np.argmax(sums_j)

def calculate_label(x, w):
    return np.sign(np.dot(x, w))

def get_learning_weights(training_set, training_labels, label):

    dimensions = len(training_set[0])
    weights = np.zeros(dimensions)
    score = 0
    score_change = True
    top_score = -1
    top_w = 0
    epochs = 0

    while score_change and epochs < NUM_EPOCHS:
        score_change = False

        for i, x in enumerate(training_set):
            y_i = 1 if training_labels[i] == label else -1
            y_hat_i = calculate_label(weights, x)

            if y_i != y_hat_i:

                # let's check if it was a good performing W
                if score > top_score:
                    score_change = True
                    top_score = score
                    top_w = np.copy(weights)

                score = 0
                weights = weights + y_i * x

            else:
                score += 1

        epochs += 1

    return top_w

def my_info():
    """
    :return: DO NOT FORGET to include your student ID as a string, this function is used to evaluate your code and results
    """
    return student_ID


def one_vs_all():
    """
    Implement the the multi label classifier using one_vs_all paradigm and return the confusion matrix
    :return: the confusion matrix regarding the result obtained using the classifier
    """

    W = [ get_learning_weights(training_set, training_labels, label) for label in range(10) ]
    W = np.matrix(W)

    predicted_labels = [predict_one_vs_all_label(x, W) for x in test_set]

    return confusion_matrix(test_labels, predicted_labels)


def all_vs_all():
    """
    Implement the multi label classifier based on the all_vs_all paradigm and return the confusion matrix
    :return: the confusing matrix obtained regarding the result obtained using teh classifier
    """
    W = [ np.zeros((10,DIMENSIONS)) for i in range(10) ]

    for i_label in range(10):
        for j_label in range(i_label + 1, 10):
            if i_label == j_label:
                continue

            # only consider two labels
            boolean_array = np.logical_or(training_labels == i_label, training_labels == j_label)
            reduced_training_labels = training_labels[boolean_array]
            reduced_training_set = training_set[boolean_array]
            wij = get_learning_weights(reduced_training_set, reduced_training_labels, i_label)
            W[i_label][j_label] = wij
            W[j_label][i_label] = -1 * wij

    predicted_labels = [predict_all_vs_all_label(x, W) for x in test_set]

    return confusion_matrix(test_labels, predicted_labels)




def main():
    """
    DO NOT TOUCH THIS FUNCTION. IT IS USED FOR COMPUTER EVALUATION OF YOUR CODE
    """
    results = my_info() + '\t\t'
    results += np.array_str(np.diagonal(one_vs_all())) + '\t\t'
    results += np.array_str(np.diagonal(all_vs_all()))
    print results + '\t\t'

if __name__ == '__main__':
    main()