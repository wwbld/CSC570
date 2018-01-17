import tensorflow as tf
import numpy as np

TRAINING = 'boston_train.csv'
TESTING = "boston_test.csv"

import util 

def deepnn(x):
    with tf.name_scope('reshape'):
        x_board = tf.reshape(x, [-1, 3, 3, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_board, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_3x3(h_conv1);

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([1*1*32, 50])
        b_fc1 = bias_variable([50])

        h_pool1_flat = tf.reshape(h_pool1, [-1, 1*1*32])
        h_fc1 = tf.matmul(h_pool1_flat, W_fc1) + b_fc1

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([50, 1])
        b_fc2 = bias_variable([1])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME')

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def main(unused_argv):
    training_data, training_target = util.read_csv(TRAINING)
    testing_data, testing_target = util.read_csv(TESTING)

    training = util.DataSet(training_data, training_target)
    test = util.DataSet(testing_data, testing_target)
   
    x = tf.placeholder(tf.float32, [None, 9], name="x")
    y_ = tf.placeholder(tf.float32, [None, 1], name="y_")

    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.square(y_conv-y_)

    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.cast(y_conv, tf.int32, name="output"),
                                      tf.cast(y_, tf.int32, name="target"))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction, name="predict_op")

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50000):
            batch = training.next_batch(50)
            if i % 1000 == 0:
                training_accuracy = cross_entropy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
                print('step %d, loss %g' % (i, training_accuracy))
            train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

        print('test loss %g' % cross_entropy.eval(feed_dict={
              x:test._images, y_:test._labels, keep_prob:1.0}))
        saver.save(sess, "model_1") 

def convertLabels(labels):
    arr = []
    for i in range(len(labels)):
        temp = [0,0,0]
        temp[int(labels[i][0])] = 1
        arr.append(temp)
    return arr


if __name__ == '__main__':
    tf.app.run()


