from collections import defaultdict
import math

import numpy

import loadmovielens

class Exercise(object):

    def __init__(self):
        super(Exercise, self).__init__()
        ratings, items_dictionary, user_ids, item_ids, movie_names = loadmovielens.read_movie_lens_data()

        self.ratings = ratings
        self.items_dictionary = items_dictionary
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.movie_names = movie_names

        self.movie_to_ratings = self.build_movie_to_ratings_dict()

    def build_movie_to_ratings_dict(self):
        movie_to_ratings = defaultdict(list)
        for rating in self.ratings:
            movie_to_ratings[rating[1]].append(rating)

        return movie_to_ratings

    def get_jaccard_coefficient_from_titles(self, title_a, title_b):
        movie_a_ids = loadmovielens.give_me_movie_id(title_a, self.items_dictionary)
        movie_b_ids = loadmovielens.give_me_movie_id(title_b, self.items_dictionary)

        if len(movie_a_ids) == 0 or len(movie_b_ids) == 0:
            exit("Movie titles not found")
        if len(movie_a_ids) > 1 or len(movie_b_ids) > 1:
            exit("Ambigous search titles")

        return self.get_jaccard_coefficient_from_ids(movie_a_ids[0][0], movie_b_ids[0][0])

    def get_jaccard_coefficient_from_ids(self, movie_a_id, movie_b_id):
        """
        :return:
        ratio: the number of users who rated both movies divided by the number of users who rated at least one of the movies.
        """

        # empty ids
        if not movie_a_id or not movie_b_id:
            return 0.0
        # movies do not exist in database
        elif movie_a_id not in self.movie_to_ratings or movie_b_id not in self.movie_to_ratings:
            return 0.0

        user_intersection = self.get_intersection_of_users(movie_a_id, movie_b_id)
        user_junction = self.get_union_of_users(movie_a_id, movie_b_id)

        return float(len(user_intersection)) / len(user_junction)

    def get_all_jaccard_coefficients_from_movie(self, movie_id):
        """
        :return:
        coefficients: array of tuples (movie_id, jaccard coeff)
        """
        if not movie_id or movie_id not in self.movie_to_ratings:
            return []

        coefficients = []
        for item_id in self.item_ids:
            coeff = self.get_jaccard_coefficient_from_ids(movie_id, item_id)
            coefficients.append((item_id, coeff))

        return coefficients

    def get_correlation_coefficient_from_titles(self, title_a, title_b):
        movie_a_ids = loadmovielens.give_me_movie_id(title_a, self.items_dictionary)
        movie_b_ids = loadmovielens.give_me_movie_id(title_b, self.items_dictionary)

        if len(movie_a_ids) == 0 or len(movie_b_ids) == 0:
            exit("Movie titles not found")
        if len(movie_a_ids) > 1 or len(movie_b_ids) > 1:
            exit("Ambigous search titles")

        return self.get_correlation_coefficient_from_ids(movie_a_ids[0][0], movie_b_ids[0][0])

    def get_correlation_coefficient_from_ids(self, movie_a_id, movie_b_id):
        # empty ids
        if not movie_a_id or not movie_b_id:
            return 0.0
        # movies do not exist in database
        elif movie_a_id not in self.movie_to_ratings or movie_b_id not in self.movie_to_ratings:
            return 0.0

        user_intersection = self.get_intersection_of_users(movie_a_id, movie_b_id)
        ratings_users_a = []

        for rating in self.movie_to_ratings[movie_a_id]:
            if rating[0] in user_intersection:
                ratings_users_a.append(rating)

        ratings_users_b = []

        for rating in self.movie_to_ratings[movie_b_id]:
            if rating[0] in user_intersection:
                ratings_users_b.append(rating)

        # we have both rating arrays, now sort them by movie ID so ratings match 1:1
        ratings_users_a = sorted(ratings_users_a, key=lambda x: x[1])
        ratings_users_b = sorted(ratings_users_b, key=lambda x: x[1])
        ratings_a = [r[2] for r in ratings_users_a]
        ratings_b = [r[2] for r in ratings_users_b]

        matrix = numpy.corrcoef(ratings_a, ratings_b)
        if len(matrix) == 0 or math.isnan(matrix[0][1]):
            return 0.0
        return matrix[0][1]

    def get_all_correlation_coefficients_from_movie(self, movie_id):
        """
        :return:
        coefficients: array of tuples (movie_id, jaccard coeff)
        """
        if not movie_id or movie_id not in self.movie_to_ratings:
            return []

        coefficients = []
        for item_id in self.item_ids:
            coeff = self.get_correlation_coefficient_from_ids(movie_id, item_id)
            coefficients.append((item_id, coeff))

        return coefficients

    def get_union_of_users(self, movie_a_id, movie_b_id):
        return self.get_users_from_ratings(self.movie_to_ratings[movie_a_id]).union(self.get_users_from_ratings(self.movie_to_ratings[movie_b_id]))

    def get_intersection_of_users(self, movie_a_id, movie_b_id):
        return self.get_users_from_ratings(self.movie_to_ratings[movie_a_id]).intersection(self.get_users_from_ratings(self.movie_to_ratings[movie_b_id]))

    def get_users_from_ratings(self, ratings):
        return set([rating[0] for rating in ratings])

    def get_movie_title_from_id(self, movie_id):
        return self.movie_names[movie_id - 1]
