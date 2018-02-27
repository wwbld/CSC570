import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from time import time
import util

TRAIN = 'iris_training.csv'
TEST = 'iris_test.csv'

batch_size = 20
num_classes = 3
epochs = 30

training_data, training_target = util.read_csv(TRAIN)
testing_data, testing_target = util.read_csv(TEST)
 
training_target = keras.utils.to_categorical(training_target, num_classes)
testing_target = keras.utils.to_categorical(testing_target, num_classes)

model = Sequential()

model.add(Reshape((2,2,1), input_shape=(4,)))
model.add(Conv2D(32, kernel_size=(2,2)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Flatten())

model.add(Dense(7))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(3))
model.add(BatchNormalization())
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(training_data, training_target,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(testing_data, testing_target),
                    callbacks=[tensorboard])
score = model.evaluate(testing_data, testing_target, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

model.save('model_1')

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("model accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
