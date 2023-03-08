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

from pre_processing import *

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

############Horovod
################################################

####Configs

#####



##############Data preparation
#Data path
data_path = pd.read_csv("../data_path.csv")
dataset_names = ["ravdess","savee","crema","tess"]
data_path_list = []
for dataset_name in dataset_names:
    df = pd.DataFrame(data_path.loc[data_path["dataset"] == dataset_name])
    data_path_list.append(df)


mappings = pd.read_csv("../new_train_mapping_fixed.csv")


##Emotion labels
emotions = ['angry','calm','disgust','fear','happy','neutral','sad','surprise']
emotion_dict = {'angry':0,'calm':1,'disgust':2,'fear':3,'happy':4,'neutral':5,'sad':6,'surprise':7}

encoder = OneHotEncoder() #One-hot-encoder object initialized to "emotion" -> [0,0,0,0,0,0,0,1]
Y = encoder.fit_transform(np.array(emotions).reshape(-1,1)).toarray()


##Job distribution
index = int(sys.argv[1])
SPLIT_PER_DATASET = 25
job_segment = int(sys.argv[1])
dataset_code = int(job_segment/SPLIT_PER_DATASET)
fragment_code = job_segment % SPLIT_PER_DATASET



    ##Current dataset
dataset = data_path_list[dataset_code] # ex) Ravdess_df
dataset_name = dataset_names[dataset_code] # ex) "ravdess"

dataset_len = len(dataset)
lines_per_dataset = math.ceil(dataset_len / SPLIT_PER_DATASET)
st = max(0, fragment_code * lines_per_dataset)
ed = min(dataset_len-1, (fragment_code+1)*lines_per_dataset)
dataset = pd.DataFrame(dataset.iloc[st:ed,:])

dataset = dataset.merge(mappings, left_on='Path', right_on='new_path',how="inner") #only training set files



####################Model Construction
model = load_model("../horovod_final/consolidated_trainable")


weights = model.get_weights()

inputs = Input(shape=(108,2048),name="consolidated_input")
# #zcr
zcr_input = Reshape(target_shape=(108,2048,1))(inputs)
zcr_x = Conv1D(16,kernel_size=16, strides=4, padding='valid', activation='relu',name = "zcr_1")(zcr_input)
zcr_x = Reshape(target_shape=(108,509*16),name="zcr_reshape")(zcr_x)
zcr_x = Dense(32,activation="relu", name="zcr_2")(zcr_x)
zcr_x = Dense(32,activation="relu", name="zcr_3")(zcr_x)
zcr_x = Dense(32,activation="relu", name="zcr_4")(zcr_x)
zcr_outputs = Dense(1,activation="relu", name="zcr_output")(zcr_x)

# #chroma
chroma_input = Reshape(target_shape=(108,2048,1))(inputs)
chroma_x = Conv1D(256,kernel_size=512, strides=1, padding='valid', activation='relu',name = "chroma_1")(chroma_input)
chroma_x = MaxPooling2D(pool_size=(1,256), strides = (1,128), padding = 'valid',name="chroma_max")(chroma_x)
chroma_x = Reshape(target_shape=(108,11*256),name="chroma_reshape")(chroma_x)
chroma_x = Dense(256,activation="relu", name="chroma_2")(chroma_x)
chroma_x = Dense(128,activation="relu", name="chroma_3")(chroma_x)
chroma_x = Dense(64,activation="relu", name="chroma_4")(chroma_x)
chroma_outputs = Dense(12,activation="relu", name="chroma"+"_output")(chroma_x)

# #rms
rms_dense = Dense(32, activation="relu", name = "rms"+"_1")
rms_x = rms_dense(inputs)
rms_x = Dense(16, activation="relu",name = "rms"+"_2")(rms_x)
rms_x = Dense(8, activation="relu",name = "rms"+"_3")(rms_x)
rms_outputs = Dense(1, activation="relu",name="rms"+"_output")(rms_x)
rms_outputs = Reshape(target_shape=(108,1))(rms_outputs)


#melspectrogram
melspectrogram_input = Reshape(target_shape=(108,2048,1))(inputs)
melspectrogram_x = Conv1D(512,kernel_size=512, strides=1, padding='valid', activation='relu',name = "melspectrogram"+"_1")(melspectrogram_input)
melspectrogram_x = MaxPooling2D(pool_size=(1,256), strides = (1,128), padding = 'valid',name="melspectrogram_max")(melspectrogram_x)
melspectrogram_x = Reshape(target_shape=(108,11*512),name="melspectrogram_reshape")(melspectrogram_x)
melspectrogram_outputs = Dense(128,activation="relu", name="melspectrogram"+"_output")(melspectrogram_x)

#mfcc
mfcc_input = Reshape(target_shape=(108,2048,1))(inputs)
mfcc_x = Conv1D(256,kernel_size=512, strides=1, padding='valid', activation='relu',name = "mfcc"+"_1")(mfcc_input)
mfcc_x = MaxPooling2D(pool_size=(1,512), strides = (1,110), padding = 'valid',name="mfcc_max")(mfcc_x)
mfcc_x = Reshape(target_shape=(108,10*256),name="mfcc_reshape")(mfcc_x)
mfcc_x = Dense(256, activation="relu",name = "mfcc"+"_2")(mfcc_x)
mfcc_x = Dense(128, activation="relu",name = "mfcc"+"_3")(mfcc_x)
mfcc_outputs = Dense(20,activation="linear", name="mfcc"+"_output")(mfcc_x)
concat = Concatenate()([zcr_outputs,chroma_outputs,mfcc_outputs,rms_outputs,melspectrogram_outputs])
feature_output = AveragePooling1D(pool_size=108)(concat)
feature_output = Reshape(target_shape=(162,1))(feature_output)

layer = BatchNormalization()(feature_output)
layer = Conv1D(256, 8, padding='same')(layer)
layer = Activation('relu')(layer)
layer = Conv1D(256, 8, padding='same')(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)
layer = Dropout(0.25)(layer)
layer = MaxPooling1D(pool_size=(8))(layer)
layer = Conv1D(128, 8, padding='same')(layer)
layer = Activation('relu')(layer)
layer = Conv1D(128, 8, padding='same')(layer)
layer = Activation('relu')(layer)
layer = Conv1D(128, 8, padding='same')(layer)
layer = Activation('relu')(layer)
layer = Conv1D(128, 8, padding='same')(layer)
layer = BatchNormalization()(layer)
layer = Activation('relu')(layer)
layer = Dropout(0.25)(layer)
layer = MaxPooling1D(pool_size=(8))(layer)
layer = Conv1D(64, 8, padding='same')(layer)
layer = Activation('relu')(layer)
layer = Conv1D(64, 8, padding='same')(layer)
layer = Activation('relu')(layer)
layer = Flatten()(layer)
layer = Dense(8)(layer) # Target class number
layer = Activation('softmax')(layer)

model = Model(inputs=inputs, outputs=layer)
utterance_model = Model(inputs=inputs, outputs=feature_output)

# #apply weights
model.compile(optimizer = "adam" , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
utterance_model.compile(optimizer = "adam" , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
print("compiled model")

model.set_weights(weights)
utterance_model.set_weights(weights[:40])
print("set model weights")



threshold = 0.8
sr = 22050
emotion_count = [0 for i in range(8)]

original_path = []
predicted_path = []


count = 0
for path, emotion in zip(dataset.file, dataset.Emotions):

    correct = 0
    # data,sr = librosa.load(path)
    #previous path: ../new_directory/train/angry_0.csv -> ../fixed_new_directory/train/angry_0.csv
    correct_path = "../fixed_new_directory/" + "/".join(path.split("/")[2:])
    try:
        data = pd.read_csv(correct_path).values.reshape(-1)
        data = divide_into_frames_collective(data)

        res = model.predict(np.array([data])) # output would be like: array([[0., 0., 0., 1., 0., 0., 0., 0.]], dtype=float32)
        audio_pred = emotions[res.argmax()] #audio_pred would be like "fear"
        if audio_pred == emotion:
            correct = 1
            
        feature = utterance_model.predict(np.array([data])).reshape(-1,)
        feature = pd.DataFrame(feature)
        
        title ="feature_approximations/"+dataset_name+"/" + emotion + "_" + str(correct) + "_" + str(fragment_code * lines_per_dataset+count) + ".csv"
        feature.to_csv(title,index=False)
        original_path.append(path)
        predicted_path.append(title)
        count+=1

        emotion_count[emotion_dict[emotion]] += 1
    except:
        print("maybe...no such file? so could not read_csv")

mappings = pd.DataFrame([])
mappings["original"] = original_path
mappings["predicted"] = predicted_path
mappings.to_csv("feature_approximation_mappings.csv",index=False, header=None, mode="a")

            
            