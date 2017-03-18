# common import for all cells
SEED=202
DATA_DIR=r'./data'
OUT_DIR =r'./out'

# standard libs
import pickle
import csv
from timeit import default_timer as timer
import os
import sys


#visualisation
%matplotlib inline

import matplotlib.pyplot as plt 
from IPython.display import Image
from IPython.display import display

# numerical libs 
import cv2
import math

import random
import numpy as np
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.set_random_seed(SEED)

from tensorflow.python.training import moving_averages
from tensorflow.contrib.framework import add_model_variable
sess = tf.InteractiveSession()

# %% 
#system("pwd")
print os.getcwd()
default_path = '/home/adioshun/Jupyter/9_개인작성/SDC/udacity-driverless-car-nd-p2/submission(notebook+html)/002'
os.chdir(default_path)

# Load pickled data  
# TODO: Fill this in based on where you saved the training and testing data

def load_data(): 
    training_file  = DATA_DIR + '/train.p'
    testing_file   = DATA_DIR + '/test.p'
    classname_file = DATA_DIR + '/signnames.csv'

    classnames = []
    with open(classname_file) as _f:
        rows = csv.reader(_f, delimiter=',')
        next(rows, None)  # skip the headers
        for i, row in enumerate(rows):
            assert(i==int(row[0]))
            classnames.append(row[1])
 
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test   = test['features'], test['labels']
    
    
    X_train  = X_train.astype(np.float32)
    y_train  = y_train.astype(np.int32)
    X_test   = X_test.astype(np.float32)
    y_test   = y_test.astype(np.int32)
    
    return  classnames, X_train, y_train, X_test, y_test 

# %% Summary
### Replace each question mark with the appropriate value.

classnames, X_train, y_train, X_test, y_test = load_data() 
 
# TODO: Number of training examples 
num_train = len(X_train)

# TODO: Number of testing examples.
num_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
_, height, width, channel = X_train.shape
image_shape = (height, width, channel)

# TODO: How many unique classes/labels there are in the dataset.
num_class = len(np.unique(y_train))


print("Number of training examples =", num_train )
print("Number of testing examples =", num_test )
print("Image data shape =", image_shape)
print("Number of classes =", num_class)

