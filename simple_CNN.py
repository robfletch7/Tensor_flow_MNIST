import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def get_CNN_model(input_shape,k=16,n_output_classes=10):
    """Function that returns a simple CNN model
    Args:
        -input_shape: shape of input images 
        -k: hyperparameter, initial number of filters"""
    model = models.Sequential()
    model.add(layers.Conv2D(filters=k
                            ,kernel_size=(3,3)
                            ,strides=(1,1)
                            ,input_shape=input_shape
                            ,padding='same'
                            ,activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    #im size 14x14
    model.add(layers.Conv2D(filters=2*k
                            ,kernel_size=(3,3)
                            ,strides=(1,1)
                            ,input_shape=input_shape
                            ,padding='same'
                            ,activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    #im size 7x7
    model.add(layers.Conv2D(filters=2*k
                            ,kernel_size=(3,3)
                            ,strides=(1,1)
                            ,input_shape=input_shape
                            ,padding='same'
                            ,activation='relu'))
    #add fully connect "Dense" layer
    model.add(layers.Flatten())
    model.add(layers.Dense(k,activation='relu'))
    model.add(layers.Dense(n_output_classes,activation='relu'))

    return model

