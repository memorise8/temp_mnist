import numpy as np
import tensorflow as tf
import pandas as pd

import csv

datasets = ['x_train', 'y_train', 'x_test', 'y_test']

with open('../data_in/x_test.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)