import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import load_model
from keras.utils import plot_model
import util

FILE = 'samples.csv'

training_data, training_target, testing_data, testing_target = util.read_csv(FILE)

model = load_model('model_1')
predictions = model.predict(testing_data)
print(predictions)

