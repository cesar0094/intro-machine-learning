import sys, ast
import loadmovielens as reader
import decimal

import exercise

"""
============================================
DO NOT FORGET TO INCLUDE YOUR STUDENT ID
============================================
"""
student_ID = '014632888'


"load the data into python"
e = exercise.Exercise()

def my_info():
    """
    :return: DO NOT FORGET to include your student ID as a string, this function is used to evaluate your code and results
    """
    return student_ID


def Jaccard_Coefficient(movie_id_1, movie_id_2):
    """
    :param movie_id_1: (integer) id regarding the first movie
    :param movie_id2: (integer) id regarding the second movie
    :return: (float) Jaccard_Coefficient regarding these movies based on the given movie IDs
            ROUND OFF TO THREE DECIMAL DIGITS
    """
    coefficient = e.get_jacobian_coefficient_from_ids(movie_id_1, movie_id_2)
    d = decimal.Decimal(coefficient)
    return round(d, 3)


def Correlation_Coefficient(movie_id_1, movie_id_2):
    """
    :param movie_id_1: (integer) id regarding the first movie
    :param movie_id2: (integer) id regarding the second movie
    :return: (float) Correlation_Coefficient regarding these movies based on the given movie IDs.
            ROUND OFF TO THREE DECIMAL DIGITS
    """
    coefficient = e.get_correlation_coefficient_from_ids(movie_id_1, movie_id_2)
    d = decimal.Decimal(coefficient)
    return round(d, 3)


def main():
    """
    DO NOT TOUCH THIS FUNCTION. IT IS USED FOR COMPUTER EVALUATION OF YOUR CODE
    """
    test_cases = ast.literal_eval(sys.argv[1])
    results = str(my_info()) + '\t\t'
    for test_case in test_cases:
        mode = test_case[0]
        id_1 = int(test_case[1])
        id_2 = int(test_case[2])
        if mode == 'jc':
            results += str(Jaccard_Coefficient(id_1, id_2)) + '\t\t'
        elif mode == 'cc':
            results += str(Correlation_Coefficient(id_1, id_2)) + '\t\t'
        else:
            exit('bad command')
    print results + '\n'

if __name__ == '__main__':
    main()
