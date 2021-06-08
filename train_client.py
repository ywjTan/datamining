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
    df = pd.read_csv('./rating_sampled.csv')
    movievecs = pd.read_csv('./movie_vec.csv', index_col=[0])
    uservecs = pd.read_csv('./user_vec.csv', index_col=[0])
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    print('processing test data set...')
    for i in range(10000 - 1000, 10000):
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
    print('processing train data set...')
    for i in range(10000-1000):
        usr = df['userId'][i]
        mv = df['movieId'][i]
        rate = df['rating'][i]
        moviev = eval(movievecs['movieVec'][mv])
        userv = eval(uservecs['userVec'][usr])
        finalv = np.array(moviev + userv)
        train_x.append(finalv)
        train_y.append(rate)
    train_x = np.array(train_x)
    # train_x = (train_x - np.mean(train_x)) / np.std(train_x)
    train_y = np.array(train_y)
    train_y = train_y.reshape([train_y.shape[0], 1])
    return {'input_train':train_x, 'label_train':train_y, 'input_test':test_x, 'label_test':test_y}


def create_placeholders():
    placeholders = []
    for var in tf.trainable_variables():
        placeholders.append(tf.placeholder_with_default(var, var.shape,
                                                        name="%s/%s" % ("FedAvg", var.op.name)))
    return placeholders


def assign_vars(local_vars, placeholders):
    reassign_ops = []
    for var, fvar in zip(local_vars, placeholders):
        reassign_ops.append(tf.assign(var, fvar))
    return tf.group(*(reassign_ops))


np.random.seed(1234)
tf.set_random_seed(1234)
PS_PUBLIC_IP = '121.36.227.50:37623'
PS_PRIVATE_IP = '121.36.227.50:37623'

communication_rounds = 20
communication = Communication(PS_PRIVATE_IP, PS_PUBLIC_IP)
client_socket = communication.start_socket_client()

print('Sending name list to the PS...')
send_message = pickle.dumps('Initial')
communication.send_message(send_message, client_socket)

print('Waiting for PS\'s command...')
sys.stdout.flush()
client_socket.settimeout(300)

received_message = pickle.loads(communication.get_message(client_socket))
hyperparameters = received_message['hyperparameters']
old_model_paras = received_message['model_paras']
local_epoch_num = hyperparameters['local_iter_num']
train_batch_size = hyperparameters['train_batch_size']
learning_rate = hyperparameters['learning_rate']
alpha = hyperparameters['alpha']

client_socket.close()

data_sets = load_data()

tf.reset_default_graph()
sess = tf.Session()
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

loss = tf.losses.mean_squared_error(net, labels_placeholder)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
saver = tf.train.Saver()
# ..............................................................................
sys.stdout.flush()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
placeholders = create_placeholders()
feed_dict = {}
for place, para in zip(placeholders, old_model_paras):
    feed_dict[place] = para
update_local_vars_op = assign_vars(tf.trainable_variables(), placeholders)
sess.run(update_local_vars_op, feed_dict=feed_dict)
print('Weights succesfully initialized')

for round_num in range(communication_rounds):
    print('Begin training')
    sys.stdout.flush()
    start_time = time.time()
    print('local epoch num = ', local_epoch_num)
    saver.save(sess, './tmp/model.ckpt')  
    for i in range(local_epoch_num):
        indices = np.random.choice(data_sets['input_train'].shape[0], train_batch_size)
        input_batch = data_sets['input_train'][indices]
        label_batch = data_sets['label_train'][indices]
        sess.run(train_step, feed_dict={input_placeholder: input_batch,
                                        labels_placeholder: label_batch})

        err = sess.run(abs(net - labels_placeholder), feed_dict={input_placeholder: input_batch,
                                                                     labels_placeholder: label_batch})
        err = np.sum(err) / train_batch_size
        print('err is: {}'.format(err))

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
    print('%d round training over' % (round_num + 1))
    print('time: %d ----> iter: %d' %
          (time.time() - start_time, local_epoch_num))
    print('')
    sys.stdout.flush()

    client_socket = communication.start_socket_client()
    client_socket.settimeout(300)
    send_message = pickle.dumps('Update')
    communication.send_message(send_message, client_socket)
    received_message = pickle.loads(communication.get_message(client_socket))
    avg_model_paras = received_message['model_paras']

    # preparing update message, delta_model_paras is a list of numpy arrays
    new_model_paras = sess.run(tf.trainable_variables())
    delta_model_paras = [np.zeros(weights.shape) for weights in new_model_paras]
    for index in range(len(new_model_paras)):
        delta_model_paras[index] = alpha*(new_model_paras[index] - avg_model_paras[index])
    for index in range(len(new_model_paras)):
        old_model_paras[index] = new_model_paras[index] - delta_model_paras[index]
    placeholders = create_placeholders()
    feed_dict = {}
    for place, para in zip(placeholders, old_model_paras):
        feed_dict[place] = para
    update_local_vars_op = assign_vars(tf.trainable_variables(), placeholders)
    sess.run(update_local_vars_op, feed_dict=feed_dict)
    print('Weights succesfully updated')
    saver.save(sess, './tmp/model.ckpt')
    send_dict = {'model_paras': delta_model_paras}
    while True:
        send_message = pickle.dumps(send_dict)
        print('begin sending trained weights')
        communication.send_message(send_message, client_socket)
        sys.stdout.flush()
        break
    print("Client trains over in round %d  and takes %f second\n" % (round_num + 1, time.time() - start_time))
    print('-----------------------------------------------------------------')
    sys.stdout.flush()
    client_socket.close()
print('finished!')
