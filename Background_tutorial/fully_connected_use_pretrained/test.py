import tensorflow as tf
import numpy as np
import util

TRAINING = 'iris_training.csv'
TESTING = "iris_test.csv"

def main():
    testing_data, testing_target = util.read_csv(TESTING)
    test = util.DataSet(testing_data, testing_target)

    graph = util.ImportGraph('./', 'model_1')
    print(graph.get_predict(test._images[0]))  


if __name__ == '__main__':
    main() 
