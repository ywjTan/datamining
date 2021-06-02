import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from communication import Communication
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_data():
    df = pd.read_csv('./data/rating.csv')
    movievecs = pd.read_csv('./movie_vec.csv', index_col=[0])
    uservecs = pd.read_csv('./user_vec.csv', index_col=[0])
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(df.shape[0]-1000):
        usr = df['userId'][i]
        mv = df['movieId'][i]
        rate = df['rating'][i]
        moviev = eval(movievecs['movieVec'][mv])
        userv = eval(uservecs['userVec'][usr])
        finalv = np.array(moviev + userv)
        train_x.append(finalv)
        train_y.append(rate)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    for i in range(df.shape[0]-1000, df.shape[0]):
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
    return {'input_train':train_x, 'label_train:':train_y, 'input_test':test_x, 'label_test':test_y}


data_sets = load_data()

learning_rate = 0.001
local_iters_num = 100000
train_batch_size = 64

tf.reset_default_graph()
sess = tf.Session()
# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, 128])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])

# Define variables
weightsl1 = tf.Variable(tf.random_normal([256, 512]))
biasesl1 = tf.Variable(tf.random_normal([512]))
weightsl2 = tf.Variable(tf.zeros([512, 128]))
biasesl2 = tf.Variable(tf.zeros([128]))
weightsl3 = tf.Variable(tf.zeros([128, 1]))
biasesl3 = tf.Variable(tf.zeros([1]))

net = images_placeholder
net = tf.nn.relu(tf.add(tf.matmul(net, weightsl1), biasesl1))
net = tf.nn.relu(tf.add(tf.matmul(net, weightsl2), biasesl2))
net = tf.add(tf.matmul(net, weightsl3), biasesl3)

loss = tf.losses.mean_squared_error(net, labels_placeholder)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
saver = tf.train.Saver()
# ..............................................................................
print('Begin training')
if os.path.exists('./tmp/checkpoint'):
    saver.restore(sess, './tmp/model.ckpt')
else:
    init = tf.global_variables_initializer()
    sess.run(init)
print('local epoch num = ', local_iters_num)
best_acc = 0
saver.save(sess, './tmp/model.ckpt')
for i in range(local_iters_num):
    indices = np.random.choice(data_sets['input_train'].shape[0], train_batch_size)
    input_batch = data_sets['input_train'][indices]
    label_batch = data_sets['label_train'][indices]
    sess.run(train_step, feed_dict={images_placeholder: input_batch,
                                    labels_placeholder: label_batch})
    err = sess.run(abs(net-labels_placeholder), feed_dict={images_placeholder: input_batch,
                                    labels_placeholder: label_batch})
    
    # if i % 10 == 0:
    #     train_accuracy = sess.run(accuracy, feed_dict={
    #         images_placeholder: input_batch, labels_placeholder: label_batch})
    #     print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))
    #     test_accuracy = sess.run(accuracy, feed_dict={
    #         images_placeholder: data_sets['images_test'],
    #         labels_placeholder: data_sets['labels_test']})
    #     print('Test accuracy {:g}'.format(test_accuracy))
    #     if best_acc < test_accuracy:
    #         best_acc = test_accuracy
    #         saver.save(sess, './tmp/model.ckpt')
    #     print('Best accuracy {:g}'.format(best_acc))
    #     with open('result.txt','w') as f:
    #         f.write('Round {}, Test accuracy {:g}\n'.format(round_num + 1, test_accuracy))
    #         f.write('Best accuracy {:g}\n'.format(best_acc))
    saver.save(sess, './tmp/model.ckpt')
