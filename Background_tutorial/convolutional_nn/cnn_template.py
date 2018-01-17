import tensorflow as tf
import numpy as np

TRAINING = 'iris_training.csv'
TESTING = "iris_test.csv"

import util 

def deepnn(x):
    with tf.name_scope('reshape'):
        x_board = tf.reshape(x, [-1, 2, 2, 1])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([2, 2, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_board, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1);

    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([1*1*32, 7])
        b_fc1 = bias_variable([7])

        h_pool1_flat = tf.reshape(h_pool1, [-1, 1*1*32])
        h_fc1 = tf.nn.softmax(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([7, 3])
        b_fc2 = bias_variable([3])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

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
   
    x = tf.placeholder(tf.float32, [None, 4], name="x")
    y_ = tf.placeholder(tf.float32, [None, 3], name="y_")

    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)

    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1, name="output"), tf.argmax(y_, 1, name="target"))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction, name="predict_op")

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = training.next_batch(20)
            arr = convertLabels(batch[1])
            if i % 1000 == 0:
                training_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:arr, keep_prob:1.0})
                print('step %d, training accuracy %g' % (i, training_accuracy))
            train_step.run(feed_dict={x:batch[0], y_:arr, keep_prob:0.5})

        arr = convertLabels(test._labels)
        print('test accuracy %g' % accuracy.eval(feed_dict={
              x:test._images, y_:arr, keep_prob:1.0}))
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


