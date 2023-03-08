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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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
from pre_processing import *

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

############Horovod
################################################


emotion_dict = {'angry':0,'calm':1,'disgust':2,'fear':3,'happy':4,'neutral':5,'sad':6,'surprise':7}
dataset_list = ["ravdess","savee","tess","crema"]
for dataset in dataset_list:

    files = sorted(glob.glob("feature_approximations/"+dataset+"/"+"*.csv"))
    feature_list = []
    emotion_list = []
    original_list = []
    emotion_count = [0 for i in range(8)]
    how_often = 10
    if dataset == "ravdess":
        how_often = 3
    elif dataset == "crema":
        how_often = 8
    elif dataset == "savee":
        how_often = 1

    for file in files: #file looks like: feature_approximations/ravdess/angry_correcness_filenum.csv
        filename = file.split("/")[-1]
        emotion = filename.split("_")[0]
        emotion_code = emotion_dict[emotion]
        
        correctness = int(filename.split("_")[1])
        filenum = int(filename.split("_")[-1].split(".")[0])
        if correctness == 1: #1 is correct
            if emotion_count[emotion_code] % how_often == 0:
                data = pd.read_csv(file).values.reshape(162,)
                feature_list.append(data)
                emotion_list.append(emotion)
                original_list.append(file)
            emotion_count[emotion_code] += 1


    X = feature_list
    X_tsne = TSNE(n_components=2,init='random', perplexity=50, random_state = 1).fit_transform(X)

    data_points = pd.DataFrame(X_tsne)
    data_points["emotion"] = emotion_list
    data_points.columns = ["x","y","emotion"]

    sp = sns.scatterplot(data=data_points,x="x",y="y",hue="emotion")
    fig = sp.get_figure()
    plt.title(dataset)
    fig.savefig("feature_plots_by_dataset/"+dataset+'_frame_scatterplot.png')
    plt.clf()

    data_points["original"] = original_list
    data_points.to_csv("feature_tsne_values/"+dataset+"_tsne_to_original_mappings.csv",index=False)