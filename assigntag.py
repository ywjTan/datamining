import pandas as pd


movie_dic = {}
movies = pd.read_csv('./data/movie.csv')
for i in range(movies.shape[0]):
    movie_dic[movies['movieId'][i]] = i
movie_tag = {}
tagscores = pd.read_csv('./data/genome_scores.csv')
for i in range(tagscores.shape[0]):
    if tagscores['relevance'][i] < 0.5:
        continue
    if tagscores['movieId'][i] not in movie_tag:
        movie_tag[tagscores['movieId'][i]] = str(tagscores['tagId'][i])
    else:
        movie_tag[tagscores['movieId'][i]] = movie_tag[tagscores['movieId'][i]] + '|' + str(tagscores['tagId'][i])
movieids = []
titles = []
genres = []
tags = []
for i in range(movies.shape[0]):
    movieids.append(movies['movieId'][i])
    titles.append(movies['title'][i])
    genres.append(movies['genres'][i])
    try:
        tags.append(movie_tag[movies['movieId'][i]])
    except KeyError:
        tags.append('Notag')
dataframe = pd.DataFrame({'movieId': movieids, 'title': titles, 'genres': genres, 'tags': tags})
dataframe.to_csv("./movie_with_tag.csv", index=False, sep=',')