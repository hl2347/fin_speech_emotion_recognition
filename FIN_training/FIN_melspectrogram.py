import pandas as pd
import numpy as np
import tensorflow as tf

import math

import os
import sys
import glob

import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from IPython.display import Audio

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint


def divide_into_frames(data,frame_length=2048, hop_length = 512):
    output = []
    n_frame = math.ceil(len(data)/hop_length)
    for i in range(n_frame-1):
        start_index = i * hop_length
        if start_index + frame_length > len(data):
            end_index = len(data)-hop_length*(n_frame-i)
            output.append(data[end_index-frame_length:end_index])
        else:
            output.append(data[start_index:start_index+frame_length])
    output.append(data[frame_length*(-1):])
    return np.array(output)

#data
# Paths for data.
print("start")
directories = sorted(glob.glob("../dataset/*/*.csv"))

file_emotion = []
frames = []
for dir in directories:
    part = dir.split('/')[-1]
    part = dir.split('_')
    file_emotion.append(part[0])
    df = pd.read_csv(dir)
    data = df.values.reshape(1,len(df))[0]
    data = divide_into_frames(data)
    frames += list(data)



input_data = np.array(frames)
input_data = input_data.reshape(len(input_data),2048)
print("reshaped input data")

mel_list = pd.read_csv("../mel_frame_labels.csv",header=None)
mel_list = mel_list.values
print("loaded mel_labels")


#Model Training
#1. Load data
mel_X = input_data
mel_y = mel_list

#mel
model=Sequential()
model.add(Conv1D(128, kernel_size=512, strides=1, padding='valid', activation='relu', input_shape=(2048, 1)))
model.add(MaxPooling1D(pool_size=512, strides = 220, padding = 'valid'))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.compile(optimizer="adam", loss="mse", metrics=["mse","mae"])

mel_x_train, mel_x_res, mel_y_train, mel_y_res = train_test_split(mel_X, mel_y,test_size=0.3,random_state=0, shuffle=True)
mel_x_val, mel_x_test, mel_y_val, mel_y_test = train_test_split(mel_x_res, mel_y_res,test_size=0.5,random_state=0, shuffle=True)
mel_x_train.shape, mel_y_train.shape, mel_x_val.shape, mel_y_val.shape, mel_x_test.shape, mel_y_test.shape

rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
history = model.fit(mel_x_train, mel_y_train, batch_size=128, epochs=50, validation_data=(mel_x_val, mel_y_val), callbacks=[rlrp])

pred = model.predict(mel_x_test)
print("r squared value: " + str(r2_score(mel_y_test, pred)))

print("MSE of our model on test data : " , model.evaluate(mel_x_test,mel_y_test)[1])

epochs = [i for i in range(50)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['mse']
train_loss = history.history['loss']
test_acc = history.history['val_mse']
test_loss = history.history['val_loss']

fig.set_size_inches(20,6)
ax[0].plot(epochs , train_loss , label = 'Training Loss')
ax[0].plot(epochs , test_loss , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()
plt.savefig("FIN_mel_training_full_dataset.png")
model.save("FIN_mel_training_full_dataset")