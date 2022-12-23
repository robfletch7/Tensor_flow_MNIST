from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import tensorflow as tf
from cv2 import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from simple_CNN import get_CNN_model
import matplotlib.pyplot as plt


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-trs','--train_split',default='0.7')
parser.add_argument('-vs','--val_split',default='0.2')
parser.add_argument('-ts','--test_split',default='0.1')
parser.add_argument('-rs','--random_seed',default='0')

args = vars(parser.parse_args())

#convert number inputs to decimal
for key in ['train_split','val_split','test_split']:
    args[key] = float(args[key])

for key in ['random_seed']:
    args[key] = int (args[key])

def Get_MNIST_data(train_split,val_split,test_split,random_seed=0):
    """Function to return MNIST dataset
    Args
        -train_split: decimal between 0 and 1 
        -val_split: decimal between 0 and 1 
        -test_split: decimal between 0 and 1 
        -random_seed: to ensure reproducible results 
    Returns:
        (X_train,y_train),(X_val,y_val),(X_test,y_test)
        """
    #load MNIST data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path='MNIST')

    #combine train and test data

    X = np.concatenate((X_train,X_test))
    y = np.concatenate((y_train,y_test))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1-train_split),random_state=random_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=((test_split/(test_split+val_split))),random_state=random_seed)
    
    return (X_train,y_train),(X_val,y_val),(X_test,y_test)

def show_random_image(images,labels,random_seed=None):
    """Function to display an image at random"""
    if random_seed:
        np.random.seed(random_seed)
    sample_index = np.random.choice(len(X_train))            
    sample_im = images[sample_index]
    sample_lab = labels[sample_index]
    sample_im = cv2.cvtColor(sample_im,cv2.COLOR_GRAY2RGB)
    cv2.imshow(str(sample_lab),sample_im)
    cv2.waitKey(0)



if __name__ == '__main__':


    #load MNIST dataset 
    (X_train,y_train),(X_val,y_val),(X_test,y_test) = Get_MNIST_data(args['train_split']
                                                                    ,args['val_split']
                                                                    ,args['test_split']
                                                                    ,random_seed=args['random_seed'])


    #create model
    model = get_CNN_model((28,28,1))
    model.summary()

    #train model
    model.compile(optimizer='adam',loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = model.fit(X_train,y_train,epochs=10,
                            validation_data = (X_val,y_val))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)


