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
from sklearn.manifold import TSNE

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

############Horovod
################################################

####Configs

#####

################################################

def preprocess_data(data):

    X = []
    res1 = adjust_length(data,2.5)
    res1 = divide_into_frames(res1)
    X.append(res1)
        
    return X
def adjust_length(data,length):
    if len(data)> 22050*length:
        midpoint = int(len(data)/2)
        left_index = midpoint - int(length*22050/2)
        return data[left_index:left_index+int(length*22050)]
    else:
        edge = int(22050*length - len(data))
        return np.pad(data,pad_width=(edge-int(edge/2),int(edge/2)),mode="constant")

def apply_weights(model,weights):
    layer_index = 0
    for layer in model.layers[1:]:
        layer.set_weights([weights[layer_index],weights[layer_index+1]])
        layer_index+=2
    return model

def divide_into_frames(data,frame_length=2048, hop_length = 512):
    output = []
    n_frame = math.ceil(len(data)/hop_length)
    for i in range(n_frame-1):
        start_index = i * hop_length
        if start_index + frame_length > len(data):
            end_index = len(data)-hop_length*(n_frame-i)
            f = data[end_index-frame_length:end_index]
            f = f.reshape(1,2048)
            output.append(f)
        else:
            f = data[start_index:start_index+frame_length]
            f = f.reshape(1,2048)
            output.append(f)
    f = data[frame_length*(-1):]
    f = f.reshape(1,2048)
    output.append(f)
    return np.array(output)

def preprocess_data_collective(data):

    X = []
    res1 = adjust_length(data,2.5)
    res1 = divide_into_frames_collective(res1)
    X.append(res1)
        
    return X

def divide_into_frames_collective(data,frame_length=2048, hop_length = 512):
    output = []
    n_frame = math.ceil(len(data)/hop_length)
    for i in range(n_frame-1):
        start_index = i * hop_length
        if start_index + frame_length > len(data):
            end_index = len(data)-hop_length*(n_frame-i)
            f = data[end_index-frame_length:end_index]
#             f = f.reshape(1,2048)
            output.append(f)
        else:
            f = data[start_index:start_index+frame_length]
#             f = f.reshape(1,2048)
            output.append(f)
    f = data[frame_length*(-1):]
#     f = f.reshape(1,2048)
    output.append(f)
    return np.array(output)


##############Data preparation
#Data path
emotions = ['angry','calm','disgust','fear','happy','neutral','sad','surprise']
for label_emotion in emotions:

    frames_df = pd.DataFrame([])
    labels = []
    peak_files = sorted(glob.glob("../meaningful_frames/*.csv"))

    for file in peak_files:
        emotion = file.split("/")[-1].split("_")[0]
        file_index = int(file.split("/")[-1].split("_")[-1].split(".")[0])
        if file_index < 50 and emotion == label_emotion:
            df = pd.DataFrame(pd.read_csv(file).iloc[:,:-2])
            label = [1 for i in range(len(df))]
            labels += label
            frames_df = pd.concat([frames_df,df])

    dip_files = sorted(glob.glob("../dipped_frames/*.csv"))

    for file in dip_files:
        emotion = file.split("/")[-1].split("_")[0]
        file_index = int(file.split("/")[-1].split("_")[-1].split(".")[0])
        if file_index < 20 and emotion == label_emotion:
            df = pd.DataFrame(pd.read_csv(file).iloc[:,:-1])
            df = pd.DataFrame(df.loc[df.iloc[:,-1] > 30])
            df = pd.DataFrame(df.iloc[:,:-1])
            label = [0 for i in range(len(df))]
            labels += label
            frames_df = pd.concat([frames_df,df])



    print(frames_df.head())
    print(len(frames_df))
    print(frames_df.values.shape)

    X = frames_df
    X_embedded = TSNE(n_components=2,init='random', perplexity=100).fit_transform(X)
    print(X_embedded.shape)

    data_points = pd.DataFrame(X_embedded)
    data_points["label"] = labels
    data_points.columns = ["x","y","label"]

    sp = sns.scatterplot(data=data_points,x="x",y="y",hue="label")
    fig = sp.get_figure()
    fig.savefig('frame_scatterplot_' + label_emotion+ '.png')
    plt.clf()
