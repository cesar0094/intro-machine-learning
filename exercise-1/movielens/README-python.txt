
Example use of Movielens data in Python:

First, import the provided package 

>>import loadmovielens as reader

Then simply use read_movie_lens_data method to read the movielens data:

>> ratings, movie_dictionary, user_ids, item_ids, movie_names = reader.read_movie_lens_data()

ratings: a numpy array where each rows is (userID, movieID, rating, timestamp)
movie_dictionary: a python dictionary where keys are integer movie_ids and values are names of the movies
user_ids: numpy array 1:943
item_ids: numpy array 1:1682
movie_names: a list of all movies in the dataset

Let's consider a rating:

>>ratings[0]
array([      196,       242,         3, 881250949])

Which means user 196 rated movie 242 as 3 with timespan 881250949

to find out what movie 242 is:
 
>>movie_dictionary[242]

'Kolya (1996)'

In order to find out the id of a movie, you can use the following method. It goes over the name of the movies and returns all the movies and the corresponding ids, if the movie contains the search term. By the way of example:

>>reader.give_me_movie_id('GoldenEye', movie_dictionary)
[(2, 'GoldenEye (1995)')]

>>reader.give_me_movie_id('story', movie_dictionary)

[(1, 'Toy Story (1995)'),
 (308, 'FairyTale: A True Story (1997)'),
 (478, 'Philadelphia Story, The (1940)'),
 (548, 'NeverEnding Story III, The (1994)'),
 (599, 'Police Story 4: Project S (Chao ji ji hua) (1993)'),
 (1072, "Pyromaniac's Love Story, A (1995)"),
 (1344, 'Story of Xinghua, The (1993)'),
 (1653, 'Entertaining Angels: The Dorothy Day Story (1996)')]



