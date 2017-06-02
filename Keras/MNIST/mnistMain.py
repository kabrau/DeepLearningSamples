import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import keras
import tensorflow as tf
sess = tf.Session()


#===============================================================================================================================
def printStepTitle(title):
    #input("PRESS ENTER TO CONTINUE.")
    print("")
    print("#----------------------------------------------------------------------------")
    print(title)
    print("#----------------------------------------------------------------------------")
    print("") 

#----------------------------------------------------------------------------
printStepTitle("# step 1 - load dataset")
#----------------------------------------------------------------------------

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)



#----------------------------------------------------------------------------
printStepTitle("# optional step - Show Images")
#----------------------------------------------------------------------------

import matplotlib.pyplot as plt

if False:
    #Color
    plt.imshow(x_train[0])
    plt.title("GroundTruth => "+str(y_train[0]))
    plt.show()

    # Gray
    plt.subplot(221) # equivalent to: plt.subplot(2, 2, 1)
    plt.title("GroundTruth => "+str(y_train[0]))
    plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))

    plt.subplot(222)
    plt.title("GroundTruth => "+str(y_train[1]))
    plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))

    plt.subplot(223)
    plt.title("GroundTruth => "+str(y_train[2]))
    plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))

    plt.subplot(224)
    plt.title("GroundTruth => "+str(y_train[3]))
    plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))

    plt.show()


#----------------------------------------------------------------------------
printStepTitle("# optional step - What's keras images shape ? channel, rows e cols OR rows, cols e channel ?")
#----------------------------------------------------------------------------

from keras import backend as K
print("image format", K.image_data_format())



#----------------------------------------------------------------------------
printStepTitle("# step 2 - input image - reshape (rows, cols => rows, cols e channel)")
#----------------------------------------------------------------------------

# input image dimensions
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print("x_train", x_train.shape)
print("x_test", x_test.shape)
print("input_shape",input_shape)



#----------------------------------------------------------------------------
printStepTitle("# step 3 - input image - change byte value to float/255")
#----------------------------------------------------------------------------

print("14 bytes before",x_train[0,10,0:13,0])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print("14 bytes after",x_train[0,10,0:13,0])




#----------------------------------------------------------------------------
printStepTitle("# step 4 - output - GroudTruth - convert class vectors to binary class matrices")
#----------------------------------------------------------------------------

print("sample 1 - before", y_train[0])

import keras

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("sample 1 - after", y_train[0])




#----------------------------------------------------------------------------
printStepTitle("# step 5 - Create or Load model")
#----------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

#Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
#MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
#Dropout(rate, noise_shape=None, seed=None)
#Flatten()
#Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

optionModel = 1

if (optionModel == 1):
    model = Sequential()
    print("layer 1 Conv2D")
    model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

elif (optionModel == 2):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

model.summary()

#----------------------------------------------------------------------------
printStepTitle("# option step - serialize model to JSON")
#----------------------------------------------------------------------------

model_json = model.to_json()
with open('./model_'+str(optionModel)+'.json', "w") as json_file:
    json_file.write(model_json)


#----------------------------------------------------------------------------
printStepTitle("# option step - Plot Model")
#----------------------------------------------------------------------------

from keras.utils.vis_utils import plot_model

plot_model(model, to_file='./model_'+str(optionModel)+'.png', show_shapes=True)  


#----------------------------------------------------------------------------
printStepTitle("# step 6 - Compile model")
#----------------------------------------------------------------------------

from keras.optimizers import RMSprop, Adadelta

#optimizer = Adadelta()
#optimizer = keras.optimizers.Adadelta()

optimizer = "Adadelta"
#optimizer = 'RMSprop'

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

print("optimizer", model.optimizer.__class__)

#----------------------------------------------------------------------------
printStepTitle("# optional step - callbacks")
#----------------------------------------------------------------------------
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, History, CSVLogger

callbacks_list = []

# checkpoint
filepath="./model_"+str(optionModel)+"_"+optimizer+"-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}.h5w"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto', save_weights_only=True)
#callbacks_list.append(checkpoint)

# earlyStopping
earlyStopping= EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
#callbacks_list.append(earlyStopping)

# History
history = False
#history = History()
#callbacks_list.append(history)

# CSVLogger
filepath="./model_"+str(optionModel)+"_"+optimizer+"_log.csv"
csv = CSVLogger(filepath, separator=',', append=False)
callbacks_list.append(csv)

#----------------------------------------------------------------------------
printStepTitle("# step 7 - Fitting")
#----------------------------------------------------------------------------
batch_size = 128
epochs = 5
#epochs = 15

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=callbacks_list)

#----------------------------------------------------------------------------
printStepTitle("# step 8 - Evaluate")
#----------------------------------------------------------------------------
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#----------------------------------------------------------------------------
printStepTitle("# optional step - Print History")
#----------------------------------------------------------------------------
if (history):
    print(history.history.keys())
    print(history.history['loss'])

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()