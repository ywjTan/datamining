import pandas as pd


df = pd.read_csv('./userinfo.csv')
movie_user = {}

for i in range(df.shape[0]):
    movies = eval(df['movies'][i])
    for movie in movies:
        if movie in movie_user:
            movie_user[movie].append(df['userId'][i])
        else:
            movie_user[movie] = []
            movie_user[movie].append(df['userId'][i])

movies = []
users = []
for movie in movie_user:
    movies.append(movie)
    users.append(movie_user[movie])
dataframe = pd.DataFrame({'movieId': movies, 'users': users})
dataframe.to_csv("./movieuser.csv", index=False, sep=',')