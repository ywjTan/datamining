import pandas as pd
import numpy as np

df = pd.read_csv('./data/rating.csv')
indices = np.random.choice(df.shape[0], 1000000)
users = []
movies = []
ratings = []
for i in range(1000000):
    users.append(df['userId'][indices[i]])
    movies.append(df['movieId'][indices[i]])
    ratings.append(df['rating'][indices[i]])
dataframe = pd.DataFrame({'userId': users, 'movieId': movies, 'rating':ratings})
dataframe.to_csv("./rating_sampled.csv", index=False, sep=',')