import sys
import numpy as np
import keras

from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed

# create a sequence classification instance
def get_sequence(n_timesteps, num_classes):
    # create a sequence of random numbers in [0,1]
    X = array([random() for _ in range(n_timesteps)])

    # calculate cut-off value to change class values
    limit2 = n_timesteps/5.0
    limit3 = n_timesteps/4.0
    limit4 = n_timesteps/3.0

    # determine the class outcome for each item in cumulative sequence
    y = []
    x = 0.0 
    for i in range(n_timesteps):
        x = x + X[i]
        value = 0
        if x < limit2:
            value = 1
        elif x < limit3:
            value = 2
        elif x < limit4:
            value = 3

        y.append(value)
        #print(x, value, limit2, limit3, limit4)

    y = np.array(y)

    # reshape input and output data to be suitable for LSTMs
    X = X.reshape(1, n_timesteps, 1)

    y = keras.utils.to_categorical(y, num_classes)
    y = y.reshape(1, n_timesteps, 4)
    return X, y

# define problem properties
n_timesteps = 10
num_classes = 4

#X,y = get_sequence(10, num_classes)
#print(X,y)
#print(X.shape)
#print(y.shape)
#sys.exit(0)

# define LSTM
model = Sequential()
model.add(LSTM(20, input_shape=(n_timesteps, 1), return_sequences=True))
model.add(TimeDistributed(Dense(num_classes, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='./model.png', show_shapes=True)  

# train LSTM
for epoch in range(10000):
    #print(epoch, "                                  ", end="")
    # generate new random sequence
    X,y = get_sequence(n_timesteps,num_classes)
    # fit model for one epoch on this sequence
    model.fit(X, y, epochs=1, batch_size=1, verbose=1)


score = model.evaluate(X, y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# evaluate LSTM
X,y = get_sequence(n_timesteps,num_classes)
yhat = model.predict_classes(X, verbose=0)
print(yhat.shape)
for i in range(n_timesteps):
    print('Expected:', y[0, i], 'Predicted', yhat[0, i])


sys.exit(0)    