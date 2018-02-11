import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import util

FILE = 'samples.csv'

batch_size = 20
num_classes = 20
epochs = 2000

training_data, training_target, testing_data, testing_target = util.read_csv(FILE)

training_target = keras.utils.to_categorical(training_target, num_classes)
testing_target = keras.utils.to_categorical(testing_target, num_classes)

model = Sequential()
model.add(Reshape((64,64,3), input_shape=(12288,)))
model.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='softmax'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(training_data, training_target,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(testing_data, testing_target))
score = model.evaluate(testing_data, testing_target, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
