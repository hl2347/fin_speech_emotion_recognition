import pandas as pd
import numpy as np
import tensorflow as tf

import os
import sys

import math

import librosa
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# from tensorflow import keras
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras import layers, Model
# from tensorflow.keras.layers import Dense,Input,Concatenate,Conv1D,MaxPooling1D,Dropout,Flatten
import keras
from keras.callbacks import ReduceLROnPlateau
from keras import layers,Model
from keras.models import load_model, Sequential
from keras.layers import Dense,Input,Concatenate,Conv1D,MaxPooling1D,Dropout,Flatten


# import keras_tuner as kt

from tensorflow.keras.utils import plot_model
print("start")
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
zcr_model=load_model("FIN_training/FIN_zcr_training_full_dataset")
chroma_model=load_model("FIN_training/FIN_chroma_training_full_dataset")
rms_model=load_model("FIN_training/FIN_rms_training_full_dataset")
melspectrogram_model=load_model("FIN_training/FIN_melspectrogram_training_full_dataset")
mfcc_model=load_model("FIN_training/FIN_mfcc_training_full_dataset")

#Model construction
inputs = Input(shape=(2048,),name="consolidated_input")

#zcr
zcr_dense = Dense(1024, activation="relu", name = "zcr"+"_1")
zcr_x = zcr_dense(inputs)
zcr_x = Dense(192, activation="tanh",name = "zcr"+"_2")(zcr_x)
zcr_x = Dense(32, activation="tanh",name = "zcr"+"_3")(zcr_x)
zcr_x = Dense(32, activation="tanh",name = "zcr"+"_4")(zcr_x)
zcr_x = Dense(32, activation="tanh",name = "zcr"+"_5")(zcr_x)
zcr_x = Dense(32, activation="tanh",name = "zcr"+"_6")(zcr_x)
zcr_x = Dense(32, activation="tanh",name = "zcr"+"_7")(zcr_x)
zcr_outputs = Dense(1, activation="sigmoid",name="zcr"+"_output")(zcr_x)
#chroma
chroma_dense = Dense(1024, activation="relu", name = "chroma"+"_1")
chroma_x = chroma_dense(inputs)
chroma_x = Dense(1024, activation="relu",name = "chroma"+"_2")(chroma_x)
chroma_x = Dense(512, activation="relu",name = "chroma"+"_3")(chroma_x)
chroma_x = Dense(256, activation="relu",name = "chroma"+"_4")(chroma_x)
chroma_x = Dense(64, activation="relu",name = "chroma"+"_5")(chroma_x)
chroma_x = Dense(32, activation="relu",name = "chroma"+"_6")(chroma_x)
chroma_x = Dense(32, activation="relu",name = "chroma"+"_7")(chroma_x)
chroma_outputs = Dense(12,activation="relu", name="chroma"+"_output")(chroma_x)
#rms
rms_dense = Dense(1024, activation="relu", name = "rms"+"_1")
rms_x = rms_dense(inputs)
rms_x = Dense(480, activation="relu",name = "rms"+"_2")(rms_x)
rms_x = Dense(704, activation="relu",name = "rms"+"_3")(rms_x)
rms_x = Dense(32, activation="relu",name = "rms"+"_4")(rms_x)
rms_x = Dense(224, activation="relu",name = "rms"+"_5")(rms_x)
rms_x = Dense(320, activation="relu",name = "rms"+"_6")(rms_x)
rms_outputs = Dense(1, activation="relu",name="rms"+"_output")(rms_x)
#melspectrogram
melspectrogram_dense = Dense(1024, activation="relu", name = "melspectrogram"+"_1")
melspectrogram_x = melspectrogram_dense(inputs)
melspectrogram_x = Dense(320, activation="relu",name = "melspectrogram"+"_2")(melspectrogram_x)
melspectrogram_x = Dense(672, activation="relu",name = "melspectrogram"+"_3")(melspectrogram_x)
melspectrogram_x = Dense(704, activation="relu",name = "melspectrogram"+"_4")(melspectrogram_x)
melspectrogram_x = Dense(640, activation="relu",name = "melspectrogram"+"_5")(melspectrogram_x)
melspectrogram_x = Dense(832, activation="relu",name = "melspectrogram"+"_6")(melspectrogram_x)
melspectrogram_outputs = Dense(128,activation="relu", name="melspectrogram"+"_output")(melspectrogram_x)
#mfcc
mfcc_dense = Dense(1024, activation="relu", name = "mfcc"+"_1")
mfcc_x = mfcc_dense(inputs)
mfcc_x = Dense(736, activation="relu",name = "mfcc"+"_2")(mfcc_x)
mfcc_x = Dense(32, activation="relu",name = "mfcc"+"_3")(mfcc_x)
mfcc_x = Dense(32, activation="relu",name = "mfcc"+"_4")(mfcc_x)
mfcc_x = Dense(32, activation="relu",name = "mfcc"+"_5")(mfcc_x)
mfcc_x = Dense(32, activation="relu",name = "mfcc"+"_6")(mfcc_x)
mfcc_x = Dense(32, activation="relu",name = "mfcc"+"_7")(mfcc_x)
mfcc_outputs = Dense(20,activation="sigmoid", name="mfcc"+"_output")(mfcc_x)
concat = Concatenate()([zcr_outputs,chroma_outputs,rms_outputs,melspectrogram_outputs,mfcc_outputs])
model = Model(inputs=inputs, outputs=concat)
concat = tf.expand_dims(concat, axis=-1)

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

#set layers untrainable
for layer in model.layers:
    layer.trainable = False
    # layer.trainable = True

#Additional layers
x = Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(concat.shape[1],1))(concat)
x = MaxPooling1D(pool_size=5, strides = 2, padding = 'same')(x)

x = Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu')(x)
x = MaxPooling1D(pool_size=5, strides = 2, padding = 'same')(x)

x = Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu')(x)
x = MaxPooling1D(pool_size=5, strides = 2, padding = 'same')(x)
x = Dropout(0.2)(x)

x = Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu')(x)
x = MaxPooling1D(pool_size=5, strides = 2, padding = 'same')(x)

x = Flatten()(x)
x = Dense(units=32, activation='relu')(x)
x = Dropout(0.3)(x)

output = Dense(units=8, activation='softmax')(x)
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Created Model")

#Data path
Ravdess = "input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
Crema = "input/cremad/AudioWAV/"
Tess = "input/Tess/Tess/"
Savee = "input/surrey-audiovisual-expressed-emotion-savee/ALL/"

#Load Dataset
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
print("loaded data")

X, Y = [], []
for path, emotion in zip(data_path.Path, data_path.Emotions):
#     file_data = load_files(path)
    data,sr = librosa.load(path)
    file_data = divide_into_frames(data)
    for ele in file_data:
        X.append(ele)
        # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
        Y.append(emotion)

# Features = pd.DataFrame(X)
# Features['labels'] = Y

# X = Features.iloc[: ,:-1].values
# Y = Features['labels'].values

# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
X = np.array(X)
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

# splitting data
x_train, x_res, y_train, y_res = train_test_split(X, Y, test_size=0.3,random_state=0, shuffle=True)
x_val, x_test, y_val, y_test = train_test_split(x_res, y_res, test_size=0.5,random_state=0, shuffle=True)
print("Before training")
rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
history=model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_val, y_val), callbacks=[rlrp])
print("trained model")
model.save("consolidated_model_untrainable_test")
print("saved model")
print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")

epochs = [i for i in range(50)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
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

# predicting on test data.
pred_test = model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)

y_test = encoder.inverse_transform(y_test)

df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = y_test.flatten()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (12, 10))
cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.savefig("consolidatd_model_test.png")
print("saved figure")