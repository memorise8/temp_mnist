import numpy as np
import pandas as pd

with np.load("../data_in/mnist.npz") as f:
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

df_x_train = pd.DataFrame(x_train)
df_y_train = pd.DataFrame(y_train)
df_x_test = pd.DataFrame(x_test)
df_y_test = pd.DataFrame(y_test)

df_x_train.to_csv('../data_in/x_train.csv', index=False)
df_y_train.to_csv('../data_in/y_train.csv', index=False)
df_x_test.to_csv('../data_in/x_test.csv', index=False)
df_y_test.to_csv('../data_in/y_test.csv', index=False)
