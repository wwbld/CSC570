import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import layers
from keras.layers.advanced_activations import LeakyReLU

def residual_block(y, channels_in, channels_out, strides=(1,1), \
                   project_shortcut=False):
    shortcut = y

    if project_shortcut == True:
        y = layers.BatchNormalization()(y)
        y = LeakyReLU()(y)
    y = layers.Conv2D(channels_in, kernel_size=(3,3), \
                      strides=strides, padding='same')(y)

    y = layers.BatchNormalization()(y)
    y = LeakyReLU()(y)
    y = layers.Conv2D(channels_out, kernel_size=(3,3), \
                      strides=(1,1), padding='same')(y)

    if project_shortcut or strides != (1,1):
        shortcut = layers.Conv2D(channels_out, kernel_size=(1,1), \
                                 strides=strides, padding='same')(shortcut)

    y = layers.add([shortcut, y]) 
 
    return y


