from __future__ import print_function
import os, sys
import pandas as pd
import json
import numpy as np

import tensorflow as tf

data_train, data_test = tf.keras.datasets.mnist.load_data()
(images_train, labels_train) = data_train
(images_test, labels_test) = data_test

# the data, split between train and test sets
# os.makedirs('./data_in', exist_ok=True)
# (x_train, y_train), (x_test, y_test) = mnist.load_data(path='/home/wonjun.sung/repository/2021/test_mnist/temp_mnist/input/data_in/mnist.npz')
#print(path)
# (x_train, y_train), (x_test, y_test) = mnist.load_data(path="../data_in/mnist.npz")
