import pandas as pd


score_data = pd.read_csv('./moviescore.csv')
movie_data = pd.read_csv('./movie_with_tag.csv')
movie_dic = {}
movie_score_list = []
for i in range(score_data.shape[0]):
    movie_dic[score_data['movieId'][i]] = i
    movie_score_list.append((score_data['movieId'][i], score_data['score'][i]))

# find the movies which have highest score
movie_score_list = sorted(movie_score_list, key=lambda kv: (kv[1], kv[0]), reverse=True)
highest1000 = movie_score_list[:1000]
for i in range(len(highest1000)):
    highest1000[i] = highest1000[i][0]
with open('./highest100.txt', 'w') as f:
    f.write(str(highest1000))

# find each genre
genre_movie = {}
for i in range(movie_data.shape[0]):
    if movie_data['movieId'][i] not in movie_dic:
        continue
    genres = movie_data['genres'][i].split('|')
    for genre in genres:
        if genre not in genre_movie:
            genre_movie[genre] = []
        genre_movie[genre].append((movie_data['movieId'][i], score_data['score'][movie_dic[movie_data['movieId'][i]]]))
genres = []
highestmovies = []
for genre in genre_movie:
    genres.append(genre)
    sortedmovies = sorted(genre_movie[genre], key=lambda kv: (kv[1], kv[0]), reverse=True)
    highestmovies.append(sortedmovies[:100])
dataframe = pd.DataFrame({'genre': genres, 'movies': highestmovies})
dataframe.to_csv("./genremovie.csv", index=False, sep=',')

# find each tag
tag_movie = {}
for i in range(movie_data.shape[0]):
    if movie_data['movieId'][i] not in movie_dic:
        continue
    if movie_data['tags'][i] == 'Notag':
        continue
    tags = movie_data['tags'][i].split('|')
    for tag in tags:
        if tag not in tag_movie:
            tag_movie[tag] = []
        tag_movie[tag].append((movie_data['movieId'][i], score_data['score'][movie_dic[movie_data['movieId'][i]]]))
tags = []
highestmovies = []
for tag in tag_movie:
    tags.append(tag)
    sortedmovies = sorted(tag_movie[tag], key=lambda kv: (kv[1], kv[0]), reverse=True)
    highestmovies.append(sortedmovies[:100])
dataframe = pd.DataFrame({'tag': tags, 'movies': highestmovies})
dataframe.to_csv("./tagmovie.csv", index=False, sep=',')