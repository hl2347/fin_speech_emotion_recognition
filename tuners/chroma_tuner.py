#import libraries
import pandas as pd
import numpy as np

import math

import librosa

from sklearn.model_selection import train_test_split


import keras
from tensorflow import keras
from keras import layers
from keras.models import Sequential, load_model
from tensorflow.keras import layers

import keras_tuner as kt

from tensorflow.keras.utils import plot_model

#function declarations
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

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(1024, input_dim=2048, activation='relu'))
    for i in range(hp.Int("num_layers", 1, 10)):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=32, max_value=1024, step=32),
                activation=hp.Choice("activation", ["relu","tanh"]),
            )
        )
    model.add(layers.Dense(12, activation="sigmoid"))
    model.compile(
        optimizer="adam", loss="mse", metrics=["mse"],
    )
    return model

#load sample audio data
sample_data = pd.read_csv("../sample_audio_data.csv")
sample_audio_data = np.array(sample_data.iloc[:,:-2])

#divide audio data into frames of equal length
input_data = []
for i in range(len(sample_audio_data)):
    input_data.append(divide_into_frames(sample_audio_data[i]))

input_data = np.array(input_data)
input_data = input_data.reshape(3650*56,2048)

build_model(kt.HyperParameters())
tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="val_mse",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="tuners",
    project_name="chroma_tuner",
)

#label input data
chroma_list = []
for i in input_data:
    stft = np.abs(librosa.stft(i,n_fft=2048, hop_length=512,center=False))
    chroma_list.append(librosa.feature.chroma_stft(S=stft,).T)

chroma_list = np.array(chroma_list)
chroma_list = chroma_list.reshape(204400,12)


#Specify Input and Output Data
chroma_X = input_data
chroma_y = chroma_list

# split data
chroma_x_train, chroma_x_rem, chroma_y_train, chroma_y_rem = train_test_split(chroma_X, chroma_y, random_state=0, shuffle=True, train_size=0.7)
chroma_x_val, chroma_x_test, chroma_y_val, chroma_y_test = train_test_split(chroma_x_rem, chroma_y_rem, random_state=0, shuffle=True, train_size=0.5)

tuner.search(chroma_x_train, chroma_y_train, epochs=50, validation_data=(chroma_x_val, chroma_y_val))

models = tuner.get_best_models(num_models=2)
best_model = models[0]
second_model = models[1]
# Build the model.
# Needed for `Sequential` without specified `input_shape`.
best_model.build(input_shape=(None, 28, 28))
second_model.build(input_shape=(None, 28, 28))

best_model.save("chroma_tuned1")
second_model.save("chroma_tuned2")