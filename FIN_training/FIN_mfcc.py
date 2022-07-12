import pandas as pd
import numpy as np
import tensorflow as tf

import math

import os
import sys

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
Ravdess = "../input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
Crema = "../input/cremad/AudioWAV/"
Tess = "../input/Tess/Tess/"
Savee = "../input/surrey-audiovisual-expressed-emotion-savee/ALL/"

ravdess_directory_list = os.listdir(Ravdess)

file_emotion = []
file_path = []
for dir in ravdess_directory_list:
    # as their are 20 different actors in our previous directory we need to extract files for each actor.
    actor = os.listdir(Ravdess + dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        # third part in each file represents the emotion associated to that file.
        file_emotion.append(int(part[2]))
        file_path.append(Ravdess + dir + '/' + file)
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

# changing integers to actual emotions.
Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)

crema_directory_list = os.listdir(Crema)

file_emotion = []
file_path = []

for file in crema_directory_list:
    # storing file paths
    file_path.append(Crema + file)
    # storing file emotions
    part=file.split('_')
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Crema_df = pd.concat([emotion_df, path_df], axis=1)

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

savee_directory_list = os.listdir(Savee)

file_emotion = []
file_path = []

for file in savee_directory_list:
    file_path.append(Savee + file)
    part = file.split('_')[1]
    ele = part[:-6]
    if ele=='a':
        file_emotion.append('angry')
    elif ele=='d':
        file_emotion.append('disgust')
    elif ele=='f':
        file_emotion.append('fear')
    elif ele=='h':
        file_emotion.append('happy')
    elif ele=='n':
        file_emotion.append('neutral')
    elif ele=='sa':
        file_emotion.append('sad')
    else:
        file_emotion.append('surprise')
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Savee_df = pd.concat([emotion_df, path_df], axis=1)

data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis = 0)

#Data preprocessing
input_data = []
for i in range(len(data_path)):
    data, sr = librosa.load(data_path.iloc[i,1],duration=2.5, offset=0.6)
    # input_data.append(divide_into_frames(data))
    input_data += list(divide_into_frames(data))
# framenum = 0
# for file in input_data:
#     framenum += len(file)

input_data = np.array(input_data)
# input_data = input_data.reshape(len(input_data)*framenum,2048)
input_data = input_data.reshape(len(input_data),2048)
print("reshaped input data")

mfcc_list = []
for i in input_data:
    mfcc_list.append(librosa.feature.mfcc(y=i,n_fft=2048,hop_length = 512,center=False).T)

mfcc_list = np.array(mfcc_list)
mfcc_list = mfcc_list.reshape(len(input_data),20)
print("reshaped mfcc list")


#Model Training
#1. Load data
mfcc_X = input_data
mfcc_y = mfcc_list

#rms model
model = Sequential()
model.add(Dense(1024,input_shape=(2048,), activation='relu', name="mfcc_1"))
model.add(Dense(736, activation='relu',name="mfcc_2"))
model.add(Dense(32, activation='relu',name="mfcc_3"))
model.add(Dense(32, activation='relu',name="mfcc_4"))
model.add(Dense(32, activation='relu',name="mfcc_5"))
model.add(Dense(32, activation='relu',name="mfcc_6"))
model.add(Dense(32, activation='relu',name="mfcc_7"))
model.add(Dense(20, activation='relu',name="output"))
model.compile(optimizer='adam',loss='mse', metrics=['mse','mae'])

mfcc_X = input_data
mfcc_y = mfcc_list
mfcc_x_train, mfcc_x_res, mfcc_y_train, mfcc_y_res = train_test_split(mfcc_X, mfcc_y,test_size=0.3,random_state=0, shuffle=True)
mfcc_x_val, mfcc_x_test, mfcc_y_val, mfcc_y_test = train_test_split(mfcc_x_res, mfcc_y_res,test_size=0.5,random_state=0, shuffle=True)


rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
history = model.fit(mfcc_x_train, mfcc_y_train, batch_size=64, epochs=50, validation_data=(mfcc_x_val, mfcc_y_val), callbacks=[rlrp])

pred = model.predict(mfcc_x_test)
print("r squared value: " + str(r2_score(mfcc_y_test, pred)))

print("MSE of our model on test data : " , model.evaluate(mfcc_x_test,mfcc_y_test)[1]*100 , "%")

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
plt.savefig("FIN_mfcc_training_full_dataset.png")
model.save("FIN_mfcc_training_full_dataset")