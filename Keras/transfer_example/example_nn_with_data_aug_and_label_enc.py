import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.preprocessing.image import ImageDataGenerator
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
from sklearn.preprocessing import LabelEncoder

FILE = 'samples.csv'

batch_size = 64
#num_classes = 9
epochs = 30
data_augmentation = True

training_data, training_target, testing_data, testing_target = util.read_csv(FILE)

training_data = np.reshape(training_data, (-1,64,64,3))
testing_data = np.reshape(testing_data, (-1,64,64,3))

# Use an encoder
encoder = LabelEncoder()
encoder.fit(training_target)
encoded_Ytrain = encoder.transform(training_target)
encoded_Ytest = encoder.transform(testing_target)

#training_target = keras.utils.to_categorical(training_target, num_classes)
#testing_target = keras.utils.to_categorical(testing_target, num_classes)

# Convert integers to dummy variables (i.e. one-hot encoded)
training_target = keras.utils.to_categorical(encoded_Ytrain)
testing_target = keras.utils.to_categorical(encoded_Ytest)
num_classes = len(training_target[0])

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

# Determine if train whole batch or partial batches
if not data_augmentation:
   print('Not using data augmentation.')
   history = model.fit(training_data, training_target,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2,
                    validation_data=(testing_data, testing_target))
else:
   print('Using real-time data augmentation.')
   # This will do preprocessing and realtime data augmentation:
   datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False) # randomly flip images
   
   # Compute quantities required for feature-wise normalization
   datagen.fit(training_data)
   
   # Fit the model on the batches generated by datagen.flow()
   model.fit_generator(datagen.flow(training_data, training_target, batch_size = batch_size),
                       epochs = epochs,
                       validation_data = (testing_data, testing_target),
                       workers = 4)
   
   
                    
# Score trained model                    
score = model.evaluate(testing_data, testing_target, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
        
total = 0
success = 0
for i in range(len(testing_target)):
    if testing_target[i] in training_target:
        total += 1
        predictions = model.predict(np.reshape(testing_data[i], (1,64,64,3)))
        if np.argmax(predictions) == np.argmax(testing_target[i]):
            success += 1
print("FINAL ACCURACY IS: ", success*1.0/total)


 
