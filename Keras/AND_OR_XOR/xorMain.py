import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import sys
import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import backend as K

print('XOR')
x_train = np.array([[0,0],[1,0],[0,1],[1,1]])
x_train = x_train.astype('float32')

y_train = np.array([0,1,1,0])

print(x_train)
print(y_train)

# create model
model = Sequential()
model.add(Dense(8, input_dim=2, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=1,
          epochs=1000,
          verbose=1)

print(model.predict_proba(x_train))

sys.exit(0)