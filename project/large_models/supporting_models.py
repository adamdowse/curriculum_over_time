import tensorflow as tf
#import tensorflow_probability as tfp
from tensorflow import keras
from keras import layers


def Simple_CNN(num_classes):
    model = tf.keras.Sequential([
        layers.Conv2D(32,(3,3), activation='relu'),
        layers.MaxPool2D((2,2)),
        layers.Flatten(),
        layers.Dense(100,activation='relu'),
        layers.Dense(num_classes,activation='softmax')
    ])
    return model

def AlexNet (num_classes):
    model = tf.keras.Sequential([
        layers.Conv2D(96, 11, strides=4, activation='relu'),
        layers.BatchNormalization(),

        layers.MaxPool2D(2, strides=2),
        
        layers.Conv2D(256,11,strides=1,activation='relu',padding='same'),
        layers.BatchNormalization(),

        layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
        layers.BatchNormalization(),
    
        layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
        layers.BatchNormalization(),

        layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu',padding="same"),
        layers.BatchNormalization(),

        layers.MaxPooling2D(2, strides=(2, 2)),

        layers.Flatten(),

        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
'''
def BCNN_AlexNet (num_classes):

    #2 conv layers and 2 dense layers
    model = tf.keras.Sequential()
    model.add(tfp.layers.Convolution2DFlipout(96, kernel_size=11, padding="same", strides=2,activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.MaxPool2D(2, strides=2))

    model.add(tfp.layers.Convolution2DFlipout(256, kernel_size=11, padding="same", strides=1,activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tfp.layers.Convolution2DFlipout(384, kernel_size=(3,3), padding="same", strides=(1,1),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tfp.layers.Convolution2DFlipout(384, kernel_size=(3,3), padding="same", strides=(1,1),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tfp.layers.Convolution2DFlipout(256, kernel_size=(3,3), padding="same", strides=(1,1),activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.MaxPooling2D(2, strides=(2, 2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tfp.python.layers.DenseFlipout(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tfp.python.layers.DenseFlipout(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tfp.python.layers.DenseFlipout(num_classes))
    return model

def BCNN_Original(num_classes):
    #2 conv layers and 2 dense layers
    model = tf.keras.Sequential()
    model.add(tfp.layers.Convolution2DFlipout(32, kernel_size=(3,3), padding="same", strides=20))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tfp.python.layers.Convolution2DFlipout(64, kernel_size=(3, 3), padding="same", strides=2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tfp.python.layers.DenseFlipout(512, activation='relu'))
    model.add(tfp.python.layers.DenseFlipout(num_classes))
    return model

'''