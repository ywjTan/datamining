import pandas as pd


df = pd.read_csv('./data/rating.csv')
movie_score = {}
for i in range(df.shape[0]):
    if df['movieId'][i] not in movie_score:
        movie_score[df['movieId'][i]] = [0, 0]
    movie_score[df['movieId'][i]][0] += df['rating'][i]
    movie_score[df['movieId'][i]][1] += 1
    if (i+1) % 100000 == 0:
        print('Processing...{}'.format((i+1)/df.shape[0]))

movies = []
scores = []

for movie in movie_score:
    movies.append(movie)
    scores.append(round((movie_score[movie][0]/movie_score[movie][1]), 2))

dataframe = pd.DataFrame({'movieId': movies, 'score': scores})
dataframe.to_csv("./moviescore.csv", index=False, sep=',')