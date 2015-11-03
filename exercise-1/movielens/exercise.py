from collections import defaultdict

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

    def get_jacobian_coefficient_from_titles(self, title_a, title_b):
        movie_a_ids = loadmovielens.give_me_movie_id(title_a, self.items_dictionary)
        movie_b_ids = loadmovielens.give_me_movie_id(title_b, self.items_dictionary)

        if len(movie_a_ids) == 0 or len(movie_b_ids) == 0:
            exit("Movie titles not found")
        if len(movie_a_ids) > 1 or len(movie_b_ids) > 1:
            exit("Ambigous search titles")

        return self.get_jacobian_coefficient_from_ids(movie_a_ids[0][0], movie_b_ids[0][0])

    def get_jacobian_coefficient_from_ids(self, movie_a_id, movie_b_id):
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

    def get_union_of_users(self, movie_a_id, movie_b_id):
        return self.get_users_from_ratings(self.movie_to_ratings[movie_a_id]).union(self.get_users_from_ratings(self.movie_to_ratings[movie_b_id]))

    def get_intersection_of_users(self, movie_a_id, movie_b_id):
        return self.get_users_from_ratings(self.movie_to_ratings[movie_a_id]).intersection(self.get_users_from_ratings(self.movie_to_ratings[movie_b_id]))

    def get_users_from_ratings(self, ratings):
        return set([rating[0] for rating in ratings])

