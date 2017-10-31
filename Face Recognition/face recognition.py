import os
import cv2
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import plot_model

#read train data
file_path='/Users/pzhe/Documents/2017_Fall/Machine Learning/1025/image/train';
os.chdir(file_path)
files = os.listdir(file_path)
h = 32;
w = 32;
x_train = np.zeros([19,1024])
for i in range(len(files)):
    img_x = cv2.imread(files[i], 0)
    img_x = img_x.astype('float32')/255
    x_train[i,:] = np.reshape(img_x,w*h,1)
y_train = np.array(range(0,19))
y_train = keras.utils.to_categorical(y_train, 19)

#read validation data
file_path='/Users/pzhe/Documents/2017_Fall/Machine Learning/1025/image/validation';
os.chdir(file_path)
files = os.listdir(file_path)
x_validation = np.zeros([19,1024])
for i in range(len(files)):
    img_x = cv2.imread(files[i], 0)
    img_x = img_x.astype('float32')/255
    x_validation[i,:] = np.reshape(img_x,w*h,1)
y_validation = np.array(range(0,19))
y_validation = keras.utils.to_categorical(y_validation, 19)

#read test data
file_path='/Users/pzhe/Documents/2017_Fall/Machine Learning/1025/image/test';
os.chdir(file_path)
files = os.listdir(file_path)
x_test = np.zeros([2,1024])
for i in range(len(files)):
    img_x = cv2.imread(files[i], 0)
    img_x = img_x.astype('float32')/255
    x_test[i,:] = np.reshape(img_x,w*h,1)
y_test = np.array([17,17])
y_test = keras.utils.to_categorical(y_test, 19)


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(1024,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(19, activation='softmax'))
model.summary()
plot_model(model, show_shapes=True,to_file='../model.png')

rmsprop=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=19,
                    epochs=200,
                    verbose=1,
                    validation_data=(x_validation, y_validation))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# plot the result
plt.plot(history.epoch,history.history['acc'],label="acc")
plt.plot(history.epoch,history.history['val_acc'],label="val_acc")
plt.scatter(history.epoch,history.history['acc'],s=10,marker='o')
plt.scatter(history.epoch,history.history['val_acc'],s=10,marker='o')
plt.legend(loc='upper left')
plt.xlim(0, 200)
plt.ylim(0.0, 1.3)
plt.title('Plot of accuracy vs. epoch(MLP)')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.epoch,history.history['loss'],label="loss")
plt.plot(history.epoch,history.history['val_loss'],label="val_loss")
plt.legend(loc='upper right')
plt.ylim(-0.1,5.0)
plt.title('Plot of loss vs. epoch(MLP)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
    
