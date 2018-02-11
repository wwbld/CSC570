import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import util

TRAIN = 'iris_training.csv'
TEST = 'iris_test.csv'

batch_size = 20
num_classes = 3
epochs = 2000

training_data, training_target = util.read_csv(TRAIN)
testing_data, testing_target = util.read_csv(TEST)
 
training_target = keras.utils.to_categorical(training_target, num_classes)
testing_target = keras.utils.to_categorical(testing_target, num_classes)

model = Sequential()
model.add(Reshape((2,2,1), input_shape=(4,)))
model.add(Conv2D(32, kernel_size=(2,2),
                 activation='relu'))
model.add(Flatten())
model.add(Dense(7, activation='softmax'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(training_data, training_target,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(testing_data, testing_target))
score = model.evaluate(testing_data, testing_target, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
