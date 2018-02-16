import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.applications import VGG16
from keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import util

FILE = 'samples.csv'

batch_size = 64
num_classes = 9
epochs = 30

training_data, training_target, testing_data, testing_target = util.read_csv(FILE)

training_data = np.reshape(training_data, (-1,64,64,3))
testing_data = np.reshape(testing_data, (-1,64,64,3))

training_target = keras.utils.to_categorical(training_target, num_classes)
testing_target = keras.utils.to_categorical(testing_target, num_classes)

vgg = VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3))
x = vgg.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=x)

for layer in vgg.layers:
    layer.trainable = False

model.compile(optimizer=optimizers.RMSprop(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(training_data, training_target,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2,
                    validation_data=(testing_data, testing_target))
score = model.evaluate(testing_data, testing_target, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
