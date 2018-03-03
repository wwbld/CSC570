import tensorflow as tf
import numpy as np
import random
import csv

from os import listdir
from os.path import isfile, isdir, join

class DataSet():
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(images)

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = [self._images[i] for i in perm]
            self._labels = [self._labels[i] for i in perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

class ImportGraph():
    def __init__(self, dire, loc):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(dire+loc+'.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(dire)) 
            self.x = self.graph.get_tensor_by_name("x:0")
            self.y_ = self.graph.get_tensor_by_name("y_:0")
            self.keep_prob = self.graph.get_tensor_by_name("dropout/keep_prob:0")
            self.output = self.graph.get_tensor_by_name("accuracy/output:0")
            self.target = self.graph.get_tensor_by_name("accuracy/target:0")
            self.predict_op = self.graph.get_tensor_by_name("predict_op:0")
 
    def get_accuracy(self, data, target):
        feed_dict = {self.x:data, self.y_:target, self.keep_prob:1.0}
        #print("accuracy is %g" % self.sess.run(self.predict_op, feed_dict))
        return self.sess.run(self.predict_op, feed_dict)

    def get_predict(self, data):
        feed_dict = {self.x:[data], self.keep_prob:1.0}
        #print("output is {0}".format(self.sess.run([self.output], feed_dict)[0][0]))
        return self.sess.run([self.output], feed_dict)[0][0]


def str2int(s):
    return int(s)

def str2float(s):
    return float(s)

# 0 means discard, 1 means train, 2 means test
# currently, 25% will be training data, 5% will be testing data
def create_sample_file():
    files = [join("./data", f) for f in listdir("./data") \
             if isfile(join("./data", f))]
    myfile = open("samples.csv", 'w')
    writer = csv.writer(myfile)
    my_list = [0]*74 + [1]*25 + [2]*1
    hashmap = {}
    for m in range(len(files)):
        with open(files[m]) as csvfile:
            temp = random.choice(my_list)
            if temp != 0:
                for line in csvfile:
                    row = line.strip().split(",")
                    row[-1] = row[-1][1:]
                    key = row[-1]
                    if hashmap.has_key(key) and hashmap.get(key) and temp == 1 < 100:
                        row.append(temp)
                        writer.writerow(row)
                    elif hashmap.has_key(key) == False and temp == 1:
                        hashmap[key] = 1
                        row.append(temp)
                        writer.writerow(row)
                    elif temp == 2:
                        row.append(temp)
                        writer.writerow(row)

def read_csv(filename):
    training_features = []
    training_labels = []
    testing_features = []
    testing_labels = []
    with open(filename) as inf:
        for line in inf:
            currentLine = line.strip().split(",")
            currentLine = list(map(str2int, currentLine))
            if currentLine[-1] == 1:
                training_features.append(currentLine[0:12288])
                training_labels.append(currentLine[12288:12289])
            else:
                testing_features.append(currentLine[0:12288])
                testing_labels.append(currentLine[12288:12289])
    return training_features, training_labels, \
           testing_features, testing_labels
