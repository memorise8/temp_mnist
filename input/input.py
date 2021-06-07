from __future__ import print_function
import os, sys
import pandas as pd
import json

import keras
from keras.datasets import mnist

# the data, split between train and test sets
os.makedirs('../data_in', exist_ok=True)
(x_train, y_train), (x_test, y_test) = mnist.load_data(path='../data_in/mnist.npz')

