import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras import backend as K
from keras import optimizers
from residual_block import residual_block
import matplotlib.pyplot as plt
import util

FILE = 'samples.csv'

batch_size = 64
num_classes = 9
epochs = 30

training_data, training_target, testing_data, testing_target = util.read_csv(FILE)

training_target = keras.utils.to_categorical(training_target, num_classes)
testing_target = keras.utils.to_categorical(testing_target, num_classes)

input = Input(shape=(12288,))
reshape = Reshape((64,64,3))(input)

# conv1
block = Conv2D(64, kernel_size=(7,7), strides=(2,2), activation='relu')(reshape)

# conv2
block = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(block)
for i in range(2):
    project_shortcut = True if i==0 else False
    block = residual_block(block, 64, 64, project_shortcut=project_shortcut)

# conv3
for i in range(2):
    strides = (2,2) if i==0 else (1,1)
    block = residual_block(block, 128, 128, strides=strides)

# conv4
for i in range(2):
    strides = (2,2) if i==0 else (1,1)
    block = residual_block(block, 256, 256, strides=strides)

# conv5
for i in range(2):
    strides = (2,2) if i==0 else (1,1)
    block = residual_block(block, 512, 512, strides=strides)
block = Activation('relu')(block)

block = GlobalAveragePooling2D()(block)
dense = Dense(num_classes)(block)
dense = BatchNormalization()(dense)
dense = Activation('softmax')(dense)

model = Model(inputs=input, outputs=dense)

rmsprop = optimizers.RMSprop(lr=0.002)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

history = model.fit(training_data, training_target,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(testing_data, testing_target))
score = model.evaluate(testing_data, testing_target, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

model.save('model_1')

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'test_acc', 'train_loss', 'test_loss'], loc='upper left')
plt.show()


