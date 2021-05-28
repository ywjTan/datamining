import pandas as pd
import numpy as np


df = pd.read_csv('./data/rating.csv')
user_movies = {}
for i in range(df.shape[0]):
    if df['userId'][i] not in user_movies:
        user_movies[df['userId'][i]] = []
    if df['rating'][i] >= 4:
        user_movies[df['userId'][i]].append(df['movieId'][i])
    if (i+1) % 100000 == 0:
        print('Processing...{}'.format((i+1)/df.shape[0]))

usr = []
mv = []
ge = []

df = pd.read_csv('./data/movie.csv')
movie_index = {}
for i in range(df.shape[0]):
    movie_index[df['movieId'][i]] = i
for user in user_movies:
    movies = user_movies[user]
    count = {}
    for movie in movies:
        genres = df['genres'][movie_index[movie]].split('|')
        for genre in genres:
            if genre not in count:
                count[genre] = 1
            else:
                count[genre] += 1
    ct = sorted(count.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    usr.append(user)
    mv.append(movies)
    if len(ct)>=3:
        ge.append([ct[0][0], ct[1][0], ct[2][0]])
    elif len(ct)>=2:
        ge.append([ct[0][0], ct[1][0]])
    elif len(ct)>=1:
        ge.append([ct[0][0]])
    else:
        ge.append([])
dataframe = pd.DataFrame({'userId': usr, 'movies': mv, 'genres': ge})
dataframe.to_csv("./userinfo.csv", index=False, sep=',')
print('end')
