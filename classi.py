import os
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_data():
    df = pd.read_csv('./rating_sampled.csv')
    movievecs = pd.read_csv('./movie_vec.csv', index_col=[0])
    uservecs = pd.read_csv('./user_vec.csv', index_col=[0])
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    print('processing test data set...')
    for i in range(100000 - 640, 100000):
        usr = df['userId'][i]
        mv = df['movieId'][i]
        rate = df['rating'][i]
        # moviev = np.array(eval(movievecs['movieVec'][mv]))
        # userv = np.array(eval(uservecs['userVec'][usr]))
        # finalv = moviev * userv
        moviev = eval(movievecs['movieVec'][mv])
        userv = eval(uservecs['userVec'][usr])
        finalv = np.array(moviev + userv)
        test_x.append(finalv)
        if rate < 3:
            test_y.append(0)
        elif rate < 3.5:
            test_y.append(1)
        elif rate < 4:
            test_y.append(2)
        elif rate < 4.5:
            test_y.append(3)
        else:
            test_y.append(4)
    test_x = np.array(test_x)
    # test_x = (test_x - np.mean(test_x)) / np.std(test_x)
    test_y = np.array(test_y)
    print('processing train data set...')
    for i in range(100000-640):
        usr = df['userId'][i]
        mv = df['movieId'][i]
        rate = df['rating'][i]
        # moviev = np.array(eval(movievecs['movieVec'][mv]))
        # userv = np.array(eval(uservecs['userVec'][usr]))
        # finalv = moviev * userv
        moviev = eval(movievecs['movieVec'][mv])
        userv = eval(uservecs['userVec'][usr])
        finalv = np.array(moviev + userv)
        train_x.append(finalv)
        if rate < 3:
            train_y.append(0)
        elif rate < 3.5:
            train_y.append(1)
        elif rate < 4:
            train_y.append(2)
        elif rate < 4.5:
            train_y.append(3)
        else:
            train_y.append(4)
    train_x = np.array(train_x)
    # train_x = (train_x - np.mean(train_x)) / np.std(train_x)
    train_y = np.array(train_y)
    count = np.zeros(5)
    for i in range(train_y.shape[0]):
        count[train_y[i]]+=1
    for i in range(5):
        print('count of {} = {}'.format(i, count[i]))
    return {'input_train':train_x, 'label_train':train_y, 'input_test':test_x, 'label_test':test_y}


data_sets = load_data()

learning_rate = 0.0001
local_iters_num = 100000
train_batch_size = 64

tf.reset_default_graph()
sess = tf.Session()
# Define input placeholders
input_placeholder = tf.placeholder(tf.float32, shape=[None, 256])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])

# Define variables
weightsl1 = tf.Variable(tf.random_normal([256, 512]))
biasesl1 = tf.Variable(tf.random_normal([512]))
weightsl2 = tf.Variable(tf.random_normal([512, 256]))
biasesl2 = tf.Variable(tf.random_normal([256]))
weightsl3 = tf.Variable(tf.random_normal([256, 128]))
biasesl3 = tf.Variable(tf.random_normal([128]))
weightsl4 = tf.Variable(tf.random_normal([128, 5]))
biasesl4 = tf.Variable(tf.random_normal([5]))

net = input_placeholder
net = tf.nn.relu(tf.add(tf.matmul(net, weightsl1), biasesl1))
net = tf.nn.relu(tf.add(tf.matmul(net, weightsl2), biasesl2))
net = tf.nn.relu(tf.add(tf.matmul(net, weightsl3), biasesl3))
net = tf.add(tf.matmul(net, weightsl4), biasesl4)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=labels_placeholder))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-8).minimize(loss)
correct_prediction = tf.equal(tf.argmax(net, 1), labels_placeholder)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
# ..............................................................................
print('Begin training')
init = tf.global_variables_initializer()
sess.run(init)
print('local epoch num = ', local_iters_num)
best_acc = 0
train_accuracy = 0

for i in range(local_iters_num):
    indices = np.random.choice(data_sets['input_train'].shape[0], train_batch_size)
    # print(indices)
    input_batch = data_sets['input_train'][indices]
    label_batch = data_sets['label_train'][indices]
    sess.run(train_step, feed_dict={input_placeholder: input_batch,
                                    labels_placeholder: label_batch})
    train_accuracy += sess.run(accuracy, feed_dict={
        input_placeholder: input_batch, labels_placeholder: label_batch})
    # print(input_batch)
    if i % 100 == 0:
        print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy/100))
        train_accuracy = 0
        # print(sess.run(tf.argmax(net,axis=1), feed_dict={input_placeholder: input_batch}))
        test_accuracy = sess.run(accuracy, feed_dict={
            input_placeholder: data_sets['input_test'],
            labels_placeholder: data_sets['label_test']})
        print('Test accuracy {:g}'.format(test_accuracy))
        # print(sess.run(tf.argmax(net, 1), feed_dict={input_placeholder: input_batch}))
        # if best_acc < test_accuracy:
        #     best_acc = test_accuracy
        #     saver.save(sess, './tmp/model.ckpt')
        # print('Best accuracy {:g}'.format(best_acc))
    #     with open('result.txt','w') as f:
    #         f.write('Round {}, Test accuracy {:g}\n'.format(round_num + 1, test_accuracy))
    #         f.write('Best accuracy {:g}\n'.format(best_acc))
    # saver.save(sess, './tmp/model.ckpt')