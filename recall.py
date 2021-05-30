import pandas as pd
import random


def simrank(users, movies, movie_dic, user):
    iters = 500
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
        if random.random() < 0.5:
            movie_start = movie_start_list[random.randint(0, set_size-1)]
            while len(eval(movies['users'][movie_dic[movie_start]])) == 0:
                movie_start = movie_start_list[random.randint(0, set_size - 1)]
    movie_visit_all = sorted(movie_visit_all.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return movie_visit_all


def recall(users, movies, movie_dic, user):
    ans = set()
    simrank_result = simrank(users, movies, movie_dic, user)
    for i in range(40):
        ans.add(simrank_result[i][0])
    print('begin tag...')
    tag_movie = pd.read_csv('./tagmovie.csv')
    tag_dic = {}
    for i in range(tag_movie.shape[0]):
        tag_dic[tag_movie['tag'][i]] = i
    usr_tags = eval(users['tags'][user-1])
    for tag in usr_tags:
        tag = int(tag)
        orig_len = len(ans)
        tag_best = eval(tag_movie['movies'][tag_dic[tag]])
        for mov, sco in tag_best:
            ans.add(mov)
            if len(ans) - orig_len >= 20:
                break
    print('begin genre...')
    ge_movie = pd.read_csv('./genremovie.csv')
    ge_dic = {}
    for i in range(ge_movie.shape[0]):
        ge_dic[ge_movie['genre'][i]] = i
    usr_ges = eval(users['genres'][user - 1])
    for ge in usr_ges:
        orig_len = len(ans)
        ge_best = eval(ge_movie['movies'][ge_dic[ge]])
        for mov, sco in ge_best:
            ans.add(mov)
            if len(ans) - orig_len >= 20:
                break
    print('begin highest...')
    with open('./highest100.txt', 'r') as f:
        hm = eval(f.read())
        for mv in hm:
            ans.add(mv)
            if len(ans) >= 200:
                break
    return ans

if __name__ == '__main__':
    movies = pd.read_csv('./movieuser.csv')
    users = pd.read_csv('./userinfo.csv')
    movie_dic = {}
    for i in range(movies.shape[0]):
        movie_dic[movies['movieId'][i]] = i
    recall_result = recall(users, movies, movie_dic, 1000)
    print(len(recall_result), recall_result)

