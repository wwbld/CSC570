import tensorflow as tf
import numpy as np
import util

TRAINING = 'boston_train.csv'
TESTING = "boston_test.csv"

def main():
    testing_data, testing_target = util.read_csv(TESTING)
    test = util.DataSet(testing_data, testing_target)

    graph = util.ImportGraph('./', 'model_1')
    
    for i in  range(20):
        print("output: {0}, target: {1}".format(graph.get_predict(test._images[i]), test._labels[i]))  


if __name__ == '__main__':
    main() 
