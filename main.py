import pandas as pd


df = pd.read_csv('./movie_vec.csv')
r = eval(df['movieVec'][0])
print('end')
