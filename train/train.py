from __future__ import print_function
import os, sys
import pandas as pd
import random
import json

import numpy as np

import tensorflow as tf
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import RMSprop

os.makedirs('./project/model', exist_ok=True)
os.makedirs('./step', exist_ok=True)

batch_size = 128
num_classes = 10
epochs = 20

target = os.environ.get('target', 'default')

if 'ipykernel' in sys.modules:
    argv = ['', '1', 'softmax', '0.1', 'False']
else:
    argv = sys.argv

try:
    epochs = int(argv[1])
except:
    epochs = 2

hidden = 512

try:
    activate = argv[2]
except:
    activate = 'softmax'

try:
    dropout = float(sys.argv[3])
except:
    dropout = 0.2

print(epochs, activate, dropout)

with np.load("../input/data_in/mnist.npz") as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']


x_train = x_train.reshape(60000, 784)
x_train = x_train[:20000]
y_train = y_train[:20000]
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('here04')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(hidden, activation='relu', input_shape=(784,)))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(hidden, activation='relu'))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(num_classes, activation=activate))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(),metrics=['accuracy'])

print('train on prd')
history = model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))

print('train on prd')

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(history.history)

arr = []
for i in range(5):
    model_path = './project/model/%s_%s.h5' % (target, "-".join(argv[1:]+[str(i)]))
    obj = {}
    obj['target'] = '%s' % (target)
    obj['args'] = " ".join(argv[1:]+[str(i)])
    obj['path'] = model_path
    obj['model'] = '%s_%s' % (target, "-".join(argv[1:]+[str(i)]))
    obj['score'] = random.random()

    model.save(model_path)

    arr.append(obj)

df = pd.DataFrame(arr)
df.to_csv('./step/train.csv', index=False)

