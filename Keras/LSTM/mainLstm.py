import sys
import numpy as np
import keras
import math

from random import choice
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed

epochs=500
samples = 200
n_timesteps = 4
opcoes = [1,2,3,4]

# max = 4*4 + 4 + 4 - 1 = 24
# min = 1*1 + 1 + 1 - 4 = -1

X = array([[choice(opcoes) for _ in range(n_timesteps)] for _ in range(samples)])
y = []
for v in X:
    y.append( (v[0]*v[1]) + v[1] + v[2] - v[3]  )

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

y = np.array(y)
y = np.reshape(y, (y.shape[0],1,1))

print(X.shape)
print(y.shape)
#sys.exit(0)

# define LSTM
model = Sequential()
model.add(LSTM(16, input_shape=(1,n_timesteps), return_sequences=True))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='./model.png', show_shapes=True)  

# train LSTM
model.fit(X, y, epochs=epochs, batch_size=20, verbose=1)

score = model.evaluate(X, y, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])

# testing
samples = 100

X = array([[choice(opcoes) for _ in range(n_timesteps)] for _ in range(samples)])
y = []
for v in X:
    y.append( (v[0]*v[1]) + v[1] + v[2] - v[3]  )

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

y = np.array(y)
y = np.reshape(y, (y.shape[0],1,1))

score = model.evaluate(X, y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#evaluate LSTM
yhat = model.predict(X, verbose=0)
#print(yhat.shape)
for i in range(samples):
    print('Expected:', y[i, 0, 0], 'Predicted', yhat[i, 0, 0])



sys.exit(0)    