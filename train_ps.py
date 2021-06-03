import tensorflow as tf
import os
import numpy as np
from communication import Communication
import pickle
tf.disable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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


PS_PRIVATE_IP = "0.0.0.0:37623"
PS_PUBLIC_IP = "0.0.0.0:37623"


local_iter_num = 100
train_batch_size = 32
learning_rate = 0.000001
alpha = 0.01

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
saver = tf.train.Saver()
if not os.path.exists('./tmp/'):
    os.mkdir('./tmp/')

if os.path.exists('./tmp/checkpoint'):
    saver.restore(sess, './tmp/model.ckpt')
else:
    init = tf.global_variables_initializer()
    sess.run(init)

communication = Communication(PS_PRIVATE_IP, PS_PUBLIC_IP)
ps_socket= communication.start_socket_ps()
ps_socket.listen(100)
hyperparameters = {'local_iter_num': local_iter_num,
                   'train_batch_size': train_batch_size,
                   'learning_rate': learning_rate,
                   'alpha': alpha}

model_paras = sess.run(tf.trainable_variables())

print("Ready for connection")
while True:
    c_1, _ = ps_socket.accept()
    # ------------------------receive-----------------------------------
    received_message_1 = pickle.loads(communication.get_message(c_1))
    if received_message_1 == 'Initial':
        send_message = {'model_paras': model_paras, 'hyperparameters': hyperparameters}
        communication.send_message(pickle.dumps(send_message), c_1)
    elif received_message_1 == 'Update':
        send_message = {'model_paras': model_paras, 'hyperparameters': hyperparameters}
        communication.send_message(pickle.dumps(send_message), c_1)

        received_message_1 = pickle.loads(communication.get_message(c_1))
        delta_model_paras_1 = received_message_1['model_paras']

        new_model_paras = [np.zeros(weights.shape) for weights in model_paras]
        print('-------------------------------')
        for index in range(len(model_paras)):
            new_model_paras[index] = model_paras[index] + delta_model_paras_1[index]
        placeholders = create_placeholders()
        feed_dict = {}
        for place, para in zip(placeholders, new_model_paras):
            feed_dict[place] = para
        update_local_vars_op = assign_vars(tf.trainable_variables(), placeholders)
        sess.run(update_local_vars_op, feed_dict=feed_dict)
        model_paras = new_model_paras
        saver.save(sess, './tmp/model.ckpt')
    c_1.close()