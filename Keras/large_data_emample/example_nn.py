import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.models import load_model
from keras import backend as K
from keras import optimizers
import util

FILE = 'samples.csv'

batch_size = 64
num_classes = 9
epochs = 2000

training_data, training_target, testing_data, testing_target = util.read_csv(FILE)

training_target = keras.utils.to_categorical(training_target, num_classes)
testing_target = keras.utils.to_categorical(testing_target, num_classes)

model = Sequential()

model.add(Reshape((64,64,3), input_shape=(12288,)))
model.add(Conv2D(32, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('softmax'))

model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(BatchNormalization())
model.add(Activation('softmax'))

rmsprop = optimizers.RMSprop(lr=0.002)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(training_data, training_target,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(testing_data, testing_target))
score = model.evaluate(testing_data, testing_target, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

model.save('model_1')
