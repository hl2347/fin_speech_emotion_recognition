import pandas as pd
import numpy as np
import tensorflow as tf

import os
import sys

import math

import librosa

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# from tensorflow import keras
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras import layers, Model
# from tensorflow.keras.layers import Dense,Input,Concatenate,Conv1D,MaxPooling1D,Dropout,Flatten
import keras
from keras import layers,Model
from keras.models import load_model, Sequential
from keras.layers import Dense,Input,Concatenate,Conv1D,MaxPooling1D,Dropout,Flatten, Lambda, Reshape


import keras_tuner as kt


# Paths for data.
Ravdess = "../input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
Crema = "../input/cremad/AudioWAV/"
Tess = "../input/Tess/Tess/"
Savee = "../input/surrey-audiovisual-expressed-emotion-savee/ALL/"

tess_directory_list = os.listdir(Tess)

file_emotion = []
file_path = []

for dir in tess_directory_list:
    directories = os.listdir(Tess + dir)
    for file in directories:
        part = file.split('.')[0]
        part = part.split('_')[2]
        if part=='ps':
            file_emotion.append('surprise')
        else:
            file_emotion.append(part)
        file_path.append(Tess + dir + '/' + file)
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Tess_df = pd.concat([emotion_df, path_df], axis=1)

data_path = Tess_df

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


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
            output.append(data[end_index-frame_length:end_index])
        else:
            output.append(data[start_index:start_index+frame_length])
    output.append(data[frame_length*(-1):])
    return np.array(output)

#Load FINs
zcr_model=load_model("../FIN/zcr")
chroma_model=load_model("../FIN/chroma")
rms_model=load_model("../FIN/rms")
melspectrogram_model=load_model("../FIN/melspectrogram")
mfcc_model=load_model("../FIN/mfcc")

#Model construction
inputs = Input(shape=(2048,),name="consolidated_input")
# #zcr
zcr_dense = Dense(1024, activation="relu", name = "zcr"+"_1")
zcr_x = zcr_dense(inputs)
zcr_x = Dense(192, activation="tanh",name = "zcr"+"_2")(zcr_x)
zcr_x = Dense(32, activation="tanh",name = "zcr"+"_3")(zcr_x)
zcr_x = Dense(32, activation="tanh",name = "zcr"+"_4")(zcr_x)
zcr_x = Dense(32, activation="tanh",name = "zcr"+"_5")(zcr_x)
zcr_x = Dense(32, activation="tanh",name = "zcr"+"_6")(zcr_x)
zcr_x = Dense(32, activation="tanh",name = "zcr"+"_7")(zcr_x)
zcr_outputs = Dense(1, activation="sigmoid",name="zcr"+"_output")(zcr_x)
# #chroma
chroma_dense = Dense(1024, activation="relu", name = "chroma"+"_1")
chroma_x = chroma_dense(inputs)
chroma_x = Dense(1024, activation="relu",name = "chroma"+"_2")(chroma_x)
chroma_x = Dense(512, activation="relu",name = "chroma"+"_3")(chroma_x)
chroma_x = Dense(256, activation="relu",name = "chroma"+"_4")(chroma_x)
chroma_x = Dense(64, activation="relu",name = "chroma"+"_5")(chroma_x)
chroma_x = Dense(32, activation="relu",name = "chroma"+"_6")(chroma_x)
chroma_x = Dense(32, activation="relu",name = "chroma"+"_7")(chroma_x)
chroma_outputs = Dense(12,activation="relu", name="chroma"+"_output")(chroma_x)
# #rms
rms_dense = Dense(1024, activation="relu", name = "rms"+"_1")
rms_x = rms_dense(inputs)
rms_x = Dense(480, activation="relu",name = "rms"+"_2")(rms_x)
rms_x = Dense(704, activation="relu",name = "rms"+"_3")(rms_x)
rms_x = Dense(32, activation="relu",name = "rms"+"_4")(rms_x)
rms_x = Dense(224, activation="relu",name = "rms"+"_5")(rms_x)
rms_x = Dense(320, activation="relu",name = "rms"+"_6")(rms_x)
rms_outputs = Dense(1, activation="relu",name="rms"+"_output")(rms_x)
#melspectrogram
melspectrogram_input = Reshape(target_shape=(2048,1))(inputs)
melspectrogram_x = Conv1D(256,kernel_size=512, strides=1, padding='valid', activation='relu',name = "melspectrogram"+"_1")(melspectrogram_input)
melspectrogram_x = MaxPooling1D(pool_size=256, strides = 110, padding = 'valid',name="melspectrogram_max")(melspectrogram_x)
melspectrogram_x = Flatten(name="melspectrogram_flat")(melspectrogram_x)
melspectrogram_outputs = Dense(128,activation="relu", name="melspectrogram"+"_output")(melspectrogram_x)
#mfcc
mfcc_input = Reshape(target_shape=(2048,1))(inputs)
mfcc_x = Conv1D(128,kernel_size=512, strides=1, padding='valid', activation='relu',name = "mfcc"+"_1")(mfcc_input)
mfcc_x = MaxPooling1D(pool_size=512, strides = 220, padding = 'valid',name="mfcc_max")(mfcc_x)
mfcc_x = Flatten(name="mfcc_flat")(mfcc_x)
mfcc_x = Dense(256, activation="relu",name = "mfcc"+"_2")(mfcc_x)
mfcc_x = Dense(128, activation="relu",name = "mfcc"+"_3")(mfcc_x)
mfcc_outputs = Dense(20,activation="linear", name="mfcc"+"_output")(mfcc_x)
concat = Concatenate()([zcr_outputs,chroma_outputs,mfcc_outputs,rms_outputs,melspectrogram_outputs])
model = Model(inputs=inputs, outputs=concat)

#weights
zcr_weights = zcr_model.get_weights()
chroma_weights = chroma_model.get_weights()
rms_weights = rms_model.get_weights()
melspectrogram_weights = melspectrogram_model.get_weights()
mfcc_weights = mfcc_model.get_weights()

#apply weights
for i in range(int(len(zcr_weights)/2)-1):
    model.get_layer("zcr_"+str(i+1)).set_weights([zcr_weights[i*2],zcr_weights[i*2+1]])
model.get_layer("zcr_output").set_weights([zcr_weights[-2],zcr_weights[-1]])
for i in range(int(len(chroma_weights)/2)-1):
    model.get_layer("chroma_"+str(i+1)).set_weights([chroma_weights[i*2],chroma_weights[i*2+1]])
model.get_layer("chroma_output").set_weights([chroma_weights[-2],chroma_weights[-1]])
for i in range(int(len(rms_weights)/2)-1):
    model.get_layer("rms_"+str(i+1)).set_weights([rms_weights[i*2],rms_weights[i*2+1]])
model.get_layer("rms_output").set_weights([rms_weights[-2],rms_weights[-1]])
for i in range(int(len(melspectrogram_weights)/2)-1):
    model.get_layer("melspectrogram_"+str(i+1)).set_weights([melspectrogram_weights[i*2],melspectrogram_weights[i*2+1]])
model.get_layer("melspectrogram_output").set_weights([melspectrogram_weights[-2],melspectrogram_weights[-1]])
for i in range(int(len(mfcc_weights)/2)-1):
    model.get_layer("mfcc_"+str(i+1)).set_weights([mfcc_weights[i*2],mfcc_weights[i*2+1]])
model.get_layer("mfcc_output").set_weights([mfcc_weights[-2],mfcc_weights[-1]])

def extract_features(data):
    # ZCR
    frames = divide_into_frames(data)
    predictions = []
    for frame in frames:
        predictions.append(model.predict(np.array([frame])))
    avg = np.mean(np.array(predictions),axis=0)
    return avg

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)
    
    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2)) # stacking vertically
    
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3)) # stacking vertically
    
    return result


X, Y = [], []
for path, emotion in zip(data_path.Path, data_path.Emotions):
    feature = get_features(path)
    for ele in feature:
        X.append(ele)
        # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
        Y.append(emotion)

Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('Tess_features.csv', index=False)