import pandas as pd
from recall import recall
import tensorflow as tf
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_data(movies, userid):
    movievecs = pd.read_csv('./movie_vec.csv', index_col=[0])
    uservecs = pd.read_csv('./user_vec.csv', index_col=[0])
    train_x = []
    print('processing train data set...')
    for i in range(len(movies)):
        usr = userid
        mv = movies[i]
        moviev = eval(movievecs['movieVec'][mv])
        userv = eval(uservecs['userVec'][usr])
        finalv = np.array(moviev + userv)
        train_x.append(finalv)
    train_x = np.array(train_x)
    return train_x


if __name__ == '__main__':
    userid = 1000
    choose_num = 20
    movies = pd.read_csv('./movieuser.csv')
    users = pd.read_csv('./userinfo.csv')
    movie_dic = {}
    for i in range(movies.shape[0]):
        movie_dic[movies['movieId'][i]] = i
    recall_result = list(recall(users, movies, movie_dic, userid))

    tf.reset_default_graph()
    sess = tf.Session()
    # Define input placeholders
    input_placeholder = tf.placeholder(tf.float32, shape=[None, 256])
    labels_placeholder = tf.placeholder(tf.float32, shape=[None, 1])

    # Define variables
    weightsl1 = tf.Variable(tf.random_normal([256, 512]))
    biasesl1 = tf.Variable(tf.random_normal([512]))
    weightsl2 = tf.Variable(tf.random_normal([512, 128]))
    biasesl2 = tf.Variable(tf.random_normal([128]))
    weightsl3 = tf.Variable(tf.random_normal([128, 1]))
    biasesl3 = tf.Variable(tf.random_normal([1]))

    net = input_placeholder
    net = tf.nn.relu(tf.add(tf.matmul(net, weightsl1), biasesl1))
    net = tf.nn.relu(tf.add(tf.matmul(net, weightsl2), biasesl2))
    net = tf.add(tf.matmul(net, weightsl3), biasesl3)

    inputs = load_data(recall_result, userid)
    saver = tf.train.Saver()
    # ..............................................................................
    print('Begin training')
    if os.path.exists('./tmp/checkpoint'):
        saver.restore(sess, './tmp/model.ckpt')
    out = sess.run(net, feed_dict={input_placeholder: inputs})
    result = []
    rec = []
    for i in range(len(recall_result)):
        result.append([recall_result[i], out[i]])
        result = sorted(result, key=lambda kv: (kv[1], kv[0]), reverse=True)
    for i in range(choose_num):
        rec.append(result[i][0])
    print(rec)

