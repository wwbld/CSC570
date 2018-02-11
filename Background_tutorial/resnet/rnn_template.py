import tensorflow as tf
import numpy as np

FILE = "samples.csv"

import util 

def build_model(x):
    with tf.name_scope('reshape'):
        x = tf.reshape(x, [-1, 64, 64, 3])

    with tf.name_scope('init'):
        x = convblock('init_conv', x, 64, 3, 16, [1,1,1,1])

    filters = [16, 16, 32, 64]

    with tf.name_scope('unit_1_0'):
        x = resblock(x, 16, 16, [1,1,1,1], True)
    for i in range(1, 5):
        with tf.name_scope('unit_1_%d' % i):
            x = resblock(x, 16, 16, [1,1,1,1], False)

    with tf.name_scope('unit_2_0'):
        x = resblock(x, 16, 32, [1,2,2,1], False)
    for i in range(1, 5):
        with tf.name_scope('unit_2_%d' % i):
            x = resblock(x, 32, 32, [1,1,1,1], False)
  
    with tf.name_scope('unit_last'):
        x = tf.nn.relu(x)
        x = tf.reduce_mean(x, [1,2])

    with tf.name_scope('fc1'):
        x = tf.reshape(x, [-1, 1*1*32])
        w = weight_variable([1*1*32, 40])
        b = bias_variable([40])
        x = tf.nn.softmax(tf.matmul(x, w) + b)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        x = tf.nn.dropout(x, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([40, 20])
        b_fc2 = bias_variable([20])
        y_conv = tf.nn.softmax(tf.matmul(x, W_fc2) + b_fc2)
 
    return y_conv, keep_prob 
  

def resblock(x, in_filter, out_filter, stride,
             activate_before_residual=False):
    if activate_before_residual:
        with tf.name_scope('shared_activation'):
            x = tf.nn.relu(x)
            orig_x = x
    else:
        with tf.name_scope('residual_activation'):
            orig_x = x
            x = tf.nn.relu(x)

    with tf.name_scope('sub1'):
        x = convblock('conv1', x, 3, in_filter, out_filter, stride)
    
    with tf.name_scope('sub2'):
        x = tf.nn.relu(x)
        x = convblock('conv2', x, 3, out_filter, out_filter, [1,1,1,1])

    with tf.name_scope('sub_add'):
        if in_filter != out_filter:
            orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
            orig_x = tf.pad(orig_x,
                     [[0,0], [0,0], [0,0],
                      [(out_filter - in_filter)//2, 
                       (out_filter - in_filter)//2]])
        x += orig_x
    
    return x



def convblock(name, x, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        kernel = weight_variable([filter_size, filter_size, in_filters, out_filters])
        return tf.nn.conv2d(x, kernel, strides, padding='SAME')

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
    training_data, training_target, testing_data, testing_target = util.read_csv(FILE)

    training = util.DataSet(training_data, training_target)
    test = util.DataSet(testing_data, testing_target)
   
    x = tf.placeholder(tf.float32, [None, 64*64*3], name="x")
    y_ = tf.placeholder(tf.float32, [None, 20], name="y_")

    y_conv, keep_prob = build_model(x)

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
            batch = training.next_batch(10)
            arr = convertLabels(batch[1])
            if i % 1 == 0:
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
        temp = [0]*20
        temp[int(labels[i][0])] = 1
        arr.append(temp)
    return arr


if __name__ == '__main__':
    tf.app.run()


