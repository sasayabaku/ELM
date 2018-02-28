# coding: utf-8

import hpelm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

digits = load_digits()

x = digits.data
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=29)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

"""
Extreme Learning Machine Architecture 

input: 64 (MNIST feature dim)
hidden: 200
output: 10 (class one-hot representation)
"""

model = hpelm.elm.ELM(x_train.shape[1], 10)
model.add_neurons(200, func='sigm')

model.train(x_train, y_train, 'c')

result = model.confusion(y_test, model.predict(x_test))

print('----- Confusion Matrix -----')
print(result)
print('----------------------------')