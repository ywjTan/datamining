import pandas as pd
import random


def simrank(users, movies, movie_dic, user):
    iters = 1000
    print("Simrank begins...")
    startmovies = set(eval(users['movies'][user-1]))
    set_size = len(startmovies)
    movie_start_list = list(startmovies)
    movie_start = movie_start_list[random.randint(0, set_size-1)]
    while movie_start not in movie_dic:
        movie_start = movie_start_list[random.randint(0, set_size - 1)]
    movie_visit_all = {}
    for it in range(iters):
        if (it+1) % 1000 == 0:
            print('Iters{}/{}'.format(it+1, iters))
        user_like = []
        movie_visit_list = []
        for us in eval(movies['users'][movie_dic[movie_start]]):
            user_like.append(us)
        user_refer = user_like[random.randint(0, len(user_like) - 1)]
        while len(eval(users['movies'][user_refer-1])) == 0:
            user_refer = user_like[random.randint(0, len(user_like) - 1)]
        for movie in eval(users['movies'][user_refer-1]):
            if movie in movie_dic:
                movie_visit_list.append(movie)
        movie_start = movie_visit_list[random.randint(0, len(movie_visit_list) - 1)]
        if movie_start in startmovies:
            continue
        elif movie_start in movie_visit_all:
            movie_visit_all[movie_start] += 1
        else:
            movie_visit_all[movie_start] = 1
        if random.random()<0.5:
            movie_start = movie_start_list[random.randint(0, set_size-1)]
            while len(eval(movies['users'][movie_dic[movie_start]])) == 0:
                movie_start = movie_start_list[random.randint(0, set_size - 1)]
    movie_visit_all = sorted(movie_visit_all.items(), key = lambda kv: (kv[1], kv[0]), reverse=True)
    return movie_visit_all


if __name__ == '__main__':
    movies = pd.read_csv('./movieuser.csv')
    users = pd.read_csv('./userinfo.csv')
    movie_dic = {}
    for i in range(movies.shape[0]):
        movie_dic[movies['movieId'][i]] = i
    simrank_result = simrank(users, movies, movie_dic, 1000)
    print(simrank_result)

