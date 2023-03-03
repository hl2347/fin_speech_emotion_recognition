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

#configs
n_epochs = 100
#data
# Paths for data.
train_data_path = sorted(glob.glob("../../new_directory/train/*.csv"))
val_data_path = sorted(glob.glob("../../new_directory/validation/*.csv"))
test_data_path = sorted(glob.glob("../../new_directory/test/*.csv"))


label_base_directory = "../../frame_features/rms/"

#Data preprocessing
train_data = []
train_label = []
for path in train_data_path:
    filename = "/".join(path.split("/")[-2:])
    data = pd.read_csv(path).values.reshape(1,-1)[0]
    train_data += list(divide_into_frames(data))
    label = list(pd.read_csv(label_base_directory+filename,header=None).values.reshape(1,-1)[0])
    train_label += label
train_data = np.array(train_data)
train_label = np.array(train_label)
print("train data: " , train_data.shape)
print("train label: " , train_label.shape)

val_data = []
val_label = []
for path in val_data_path:
    filename = "/".join(path.split("/")[-2:])
    data = pd.read_csv(path).values.reshape(1,-1)[0]
    val_data += list(divide_into_frames(data))
    label = list(pd.read_csv(label_base_directory+filename,header=None).values.reshape(1,-1)[0])
    val_label += label
val_data = np.array(val_data)
val_label = np.array(val_label)
print("validation data: " , val_data.shape)
print("validation label: " , val_label.shape)

test_data = []
test_label = []
for path in test_data_path:
    filename = "/".join(path.split("/")[-2:])
    data = pd.read_csv(path).values.reshape(1,-1)[0]
    test_data += list(divide_into_frames(data))
    label = list(pd.read_csv(label_base_directory+filename,header=None).values.reshape(1,-1)[0])
    test_label += label
test_data = np.array(test_data)
test_label = np.array(test_label)
print("test data: " , test_data.shape)
print("test label: " , test_label.shape)

#rms model
model = Sequential()
model.add(Dense(32,input_shape=(2048,), activation='relu', name="rms_1"))
model.add(Dense(16, activation='relu',name="rms_2"))
model.add(Dense(8, activation='relu',name="rms_3"))
model.add(Dense(1, activation='relu',name="output"))
model.compile(optimizer='adam',loss='mse', metrics=['mse','mae'])


rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
history = model.fit(train_data, train_label, batch_size=64, epochs=n_epochs, validation_data=(val_data, val_label), callbacks=[rlrp])

pred = model.predict(test_data)
print("r squared value: " + str(r2_score(test_label, pred)))

print("MSE of our model on test data : " , model.evaluate(test_data,test_label)[1]*100 , "%")

epochs = [i for i in range(n_epochs)]
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
plt.savefig("rms.png")
model.save("rmss")