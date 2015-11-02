__author__ = 'sorkhei'

import os
import numpy as np


def read_movie_lens_data():
    """
    :return:
    ratings: a numpy array where each rows is (userID, itemID, rating, timestamp)
    movie_dictionary: a python dictionary where keys are integer movie_ids and values are names of the movies
    userids: numpy array 1:943
    itemids: numpy array 1:1682
    movie_names: a list of all movies in data set
    """
    if 'data' not in os.listdir(os.getcwd()):
        exit('data directory not found')

    data_dir = os.path.join(os.getcwd(), 'data')
    data_dir_content = os.listdir(data_dir)

    if 'u.data' not in data_dir_content or 'u.item' not in data_dir_content:
        exit('required files not found')

    ratings = np.loadtxt(os.path.join(data_dir, 'u.data'), dtype=np.int)

    items = open(os.path.join(data_dir, 'u.item')).readlines()
    items_dictionary = dict([(int(item.split('|')[0]), item.split('|')[1]) for item in items])
    movie_names = items_dictionary.values()
    user_ids = np.arange(1, 943)
    item_ids = np.arange(1, 1682)

    return ratings, items_dictionary, user_ids, item_ids, movie_names

def give_me_movie_id(movie_name, movie_dictionary):
    """
    returns a list of movies and movie ids if possible
    """

    found_movies = [item for item in movie_dictionary.items() if movie_name.lower() in item[1].lower()]
    return found_movies
