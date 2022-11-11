import pandas as pd
import numpy as np

import os
import sys
import glob

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
import math
import random
import time

from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# to play the audio files
from IPython.display import Audio

import keras
from keras import Model
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Activation, Input, MaxPooling2D, Reshape, Concatenate,AveragePooling1D
# from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence, to_categorical
import horovod.tensorflow as hvd
import tensorflow as tf
import horovod.keras.callbacks as hvdk


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

def oob(test, pred, sensitivity):
    hit = 0
    total = len(test) * len(test[0])
    for x in range(len(test)):
        result = np.absolute(test[x]-pred[x])
        result = np.where(result <= sensitivity, 1, 0).sum()
        hit += result
    return hit/total


#load model
model=load_model("FIN_melspectrogram_training_full_dataset")

#data
# Paths for data.
data_path = sorted(glob.glob("../../new_dataset_csv/*.csv"))
data_length = int(len(data_path) / 3)


label_base_directory = "../../new_dataset_frame_features/melspectrogram/"

#Data preprocessing
train_data = []
train_label = []
for path in data_path[:data_length]:
    filename = path.split("/")[-1]
    data = pd.read_csv(path).values.reshape(1,-1)[0]
    train_data += list(divide_into_frames(data))
    label = list(pd.read_csv(label_base_directory+filename,header=None).values)
    train_label += label
train_data = np.array(train_data)
train_label = np.array(train_label)
print("train data: " , train_data.shape)
print("train label: " , train_label.shape)


print("MSE of our model on test data : " , model.evaluate(train_data,train_label)[1]*100 , "%")
pred = model.predict(train_data)
print("r squared value: " + str(r2_score(train_label, pred)))
print("r squared value var weighted: " + str(r2_score(train_label, pred, multioutput='variance_weighted')))
print("oob 0.01: "+  str(oob(train_label,pred,0.01)))
print("oob 0.005: "+  str(oob(train_label,pred,0.005)))
print("oob 0.001: "+  str(oob(train_label,pred,0.001)))
print("oob 0.0001: "+  str(oob(train_label,pred,0.0001)))
print("oob 0.00001: "+  str(oob(train_label,pred,0.00001)))