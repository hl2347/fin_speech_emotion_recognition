import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# to play the audio files
from IPython.display import Audio

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization,Activation
# from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint
from pre_processing import *

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


#Data path
data_path = pd.read_csv("data_path.csv") #columns: Emotions, Path, dataset

#For each dataset
dataset_names = ["ravdess","savee","tess","crema"]
dataset_list = []
counter = [[0 for i in range(8)] for x in range(3)]
for name in dataset_names:
    df = pd.DataFrame(data_path.loc[data_path["dataset"] == name])
    #8 emotions
    emotion_list = ['disgust', 'neutral', 'fear', 'calm', 'happy', 'angry', 'surprise', 'sad']
    for emotion_index, emotion in enumerate(emotion_list):
        emotion_df = df.loc[df.Emotions == emotion]
        emotion_df = emotion_df.sample(frac=1,random_state = 0)
        train_index = int(len(emotion_df) * 0.7)
        validation_index = train_index+int(len(emotion_df)*0.15)

        train_df = emotion_df.iloc[:train_index,:]
        val_df = emotion_df.iloc[train_index:validation_index,:]
        test_df = emotion_df.iloc[validation_index:,:]
        
        for i in range(len(train_df)):
            data,sr = librosa.load(train_df.iloc[i,1],duration=2.5, offset=0.6)
            data = adjust_length(data,2.5)
            data = pd.DataFrame(data)
            data.to_csv("fixed_new_directory/train/"+emotion+"_"+str(counter[0][emotion_index])+".csv",index=False)
            counter[0][emotion_index] += 1

        for i in range(len(val_df)):
            data,sr = librosa.load(val_df.iloc[i,1],duration=2.5, offset=0.6)
            data = adjust_length(data,2.5)
            data = pd.DataFrame(data)
            data.to_csv("fixed_new_directory/validation/"+emotion+"_"+str(counter[1][emotion_index])+".csv",index=False)
            counter[1][emotion_index] += 1

        for i in range(len(test_df)):
            data,sr = librosa.load(test_df.iloc[i,1],duration=2.5, offset=0.6)
            data = adjust_length(data,2.5)
            data = pd.DataFrame(data)
            data.to_csv("fixed_new_directory/test/"+emotion+"_"+str(counter[2][emotion_index])+".csv",index=False)
            counter[2][emotion_index] += 1