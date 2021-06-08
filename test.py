import os
import numpy as np
import pandas as pd
import tensorflow as tf
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_data():
    df = pd.read_csv('./rating_test.csv')
    movievecs = pd.read_csv('./movie_vec.csv', index_col=[0])
    uservecs = pd.read_csv('./user_vec1.csv', index_col=[0])
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    print('processing test data set...')
    for i in range(df.shape[0]):
        usr = df['userId'][i]
        mv = df['movieId'][i]
        rate = df['rating'][i]
        moviev = eval(movievecs['movieVec'][mv])
        userv = eval(uservecs['userVec'][usr])
        finalv = np.array(moviev + userv)
        test_x.append(finalv)
        test_y.append(rate)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    test_y = test_y.reshape([test_y.shape[0], 1])
    # print('processing train data set...')
    # for i in range(500000-1000):
    #     usr = df['userId'][i]
    #     mv = df['movieId'][i]
    #     rate = df['rating'][i]
    #     moviev = eval(movievecs['movieVec'][mv])
    #     userv = eval(uservecs['userVec'][usr])
    #     finalv = np.array(moviev + userv)
    #     train_x.append(finalv)
    #     train_y.append(rate)
    # train_x = np.array(train_x)
    # # train_x = (train_x - np.mean(train_x)) / np.std(train_x)
    # train_y = np.array(train_y)
    # train_y = train_y.reshape([train_y.shape[0], 1])
    return {'input_train':train_x, 'label_train':train_y, 'input_test':test_x, 'label_test':test_y}


data_sets = load_data()

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

saver = tf.train.Saver()
if os.path.exists('./tmp/checkpoint'):
    saver.restore(sess, './tmp/model.ckpt')
# ..............................................................................
print('Begin test')

err = sess.run(abs(net-labels_placeholder), feed_dict={input_placeholder: data_sets['input_test'], labels_placeholder: data_sets['label_test']})
err = np.sum(err)/56
print('error {}'.format(round(err, 3)))
