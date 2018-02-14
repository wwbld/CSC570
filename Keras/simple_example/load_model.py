import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import load_model
from keras.utils import plot_model
import numpy
import util

TRAIN = 'iris_training.csv'
TEST = 'iris_test.csv'

training_data, training_target = util.read_csv(TRAIN)
testing_data, testing_target = util.read_csv(TEST)

model = load_model('model_1')
predictions = model.predict(testing_data)
for i in range(len(predictions)):
    print("{0:3d}: {1}  {2}".format(i, predictions[i], testing_target[i]))
    # sort the prediction in descending order, return there indices.
    #print("sorted: {0}".format(numpy.argsort(predictions[i])[::-1]))

