import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import load_model
from keras.utils import plot_model

model = load_model('model_1')
plot_model(model, to_file='model_1.png')

