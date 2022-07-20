import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
import math
from random import shuffle 

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

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(130,2048), n_channels=1,
                 n_classes=8, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # Generate data
        X = []
        y = []
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
#             X[i,] = np.load('data/' + ID + '.npy')

            data, sample_rate = librosa.load(ID, duration=2.5, offset=0.6)
            X += preprocess_data(data) 


            # Store class
            label = self.labels[ID]
            for i in range(3):
                y.append(to_categorical(label, num_classes=self.n_classes))
        print(np.array(X).shape)
        return np.array(X), np.array(y)

#Data path
Ravdess = "../input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
Crema = "../input/cremad/AudioWAV/"
Tess = "../input/Tess/Tess/"
Savee = "../input/surrey-audiovisual-expressed-emotion-savee/ALL/"

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

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(list(set(data_path.Emotions))).reshape(-1,1)).toarray()

#Labeling for data generator
labels = {}
emotions = ['angry','calm','disgust','fear','happy','neutral','sad','surprise']
print("loaded data path")


sample_rate = 22050
i = 0 
for emotion in emotions:
    paths = data_path.loc[data_path["Emotions"] == emotion].Path
    for path in paths:
        labels[path] = i
    i+=1 


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

def adjust_length(data,length):
    if len(data)> 22050*length:
        midpoint = int(len(data)/2)
        left_index = midpoint - int(length*22050/2)
        return data[left_index:left_index+length*22050]
    else:
        edge = 22050*length - len(data)
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
            output.append(data[end_index-frame_length:end_index])
        else:
            output.append(data[start_index:start_index+frame_length])
    output.append(data[frame_length*(-1):])
    return np.array(output)



def preprocess_data(data):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    X = []
    # without augmentation
    res1 = adjust_length(data,3)
    res1 = divide_into_frames(res1)
    X.append(res1)
    
    # data with noise
    noise_data = noise(data)
    res2 = adjust_length(noise_data,3)
    res2 = divide_into_frames(res2)
    X.append(res2)
    
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = adjust_length(data_stretch_pitch,3)
    res3 = divide_into_frames(res3)
    X.append(res3)
    
    return X


# def custom_generator(input_ids, batch_size = 32):
  
#   while True:
#     batch_paths = np.random.choice(a= input_ids, size = batch_size)
    
#     batch_input = []
#     batch_output = []
    
#     for input_id in batch_paths:
#         data, sample_rate = librosa.load(input_id, duration=2.5, offset=0.6)
#         input = data
#         label = labels[input_id]
#         output = [0 for x in range(8)]
#         output[label] = 1
      
#         input = preprocess_data(input)

#         batch_input += input
#         for i in range(3):
#             batch_output.append(output)
   
#         batch_x = np.array(batch_input)
#         batch_y = np.array(batch_output)

    
#     yield (batch_x, batch_y)

# batch_size = 8

img_ids = list(labels.keys())
shuffle(img_ids)

split = int(0.7 * len(img_ids))
split2 = int(0.85 * len(img_ids))

train_ids = img_ids[0:split]
valid_ids = img_ids[split:split2]
test_ids = img_ids[split2:]

# train_generator = custom_generator(train_ids, batch_size = batch_size)
# valid_generator = custom_generator(valid_ids, batch_size = batch_size)
# test_generator = custom_generator(test_ids, batch_size = batch_size)

params = {'dim': (130,2048),
          'batch_size': 8,
          'n_classes': 8,
          'n_channels': 1,
          'shuffle': False}

partition = {'train':train_ids, 'validation':valid_ids,'test':test_ids}

training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)
test_generator = DataGenerator(partition['test'], labels, **params)

y_test = []
for p in test_ids:
     y_test.append(emotions[labels[p]])
y_test = encoder.transform(np.array(y_test).reshape(-1,1)).toarray()


#Load FINs
zcr_model=load_model("../FIN/zcr")
chroma_model=load_model("../FIN/chroma")
rms_model=load_model("../FIN/rms")
melspectrogram_model=load_model("../FIN/melspectrogram")
mfcc_model=load_model("../FIN/mfcc")
print("loaded models")


#weights
zcr_weights = zcr_model.get_weights()
chroma_weights = chroma_model.get_weights()
rms_weights = rms_model.get_weights()
melspectrogram_weights = melspectrogram_model.get_weights()
mfcc_weights = mfcc_model.get_weights()

print("loaded model weights")

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
    X = []
    # without augmentation
    res1 = adjust_length(data,3)
    res1 = divide_into_frames(res1)
    X.append(res1)
    
    # data with noise
    noise_data = noise(data)
    res2 = adjust_length(noise_data,3)
    res2 = divide_into_frames(res2)
    X.append(res2)
    
    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = adjust_length(data_stretch_pitch,3)
    res3 = divide_into_frames(res3)
    X.append(res3)
    
    return np.array(X)


# X, Y = [], []
# for path, emotion in zip(data_path.Path, data_path.Emotions):
#     feature = get_features(path)
#     for ele in feature:
#         X.append(ele)
#         # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
#         Y.append(emotion)

# Features = pd.DataFrame(X)
# Features['labels'] = Y


# As this is a multiclass classification problem onehotencoding our Y.


# splitting data
# x_train, x_res, y_train, y_res = train_test_split(X, Y, test_size=0.3,random_state=0, shuffle=True)
# x_val, x_test, y_val, y_test = train_test_split(x_res, y_res, test_size=0.5,random_state=0, shuffle=True)

print("prepared X,y")

#Model construction
inputs = Input(shape=(130,2048),name="consolidated_input")
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
melspectrogram_input = Reshape(target_shape=(130,2048,1))(inputs)
melspectrogram_x = Conv1D(256,kernel_size=512, strides=1, padding='valid', activation='relu',name = "melspectrogram"+"_1")(melspectrogram_input)
melspectrogram_x = MaxPooling2D(pool_size=(1,256), strides = (1,110), padding = 'valid',name="melspectrogram_max")(melspectrogram_x)
# melspectrogram_x = Flatten(name="melspectrogram_flat")(melspectrogram_x)
melspectrogram_x = Reshape(target_shape=(130,12*256),name="melspectrogram_reshape")(melspectrogram_x)
melspectrogram_outputs = Dense(128,activation="relu", name="melspectrogram"+"_output")(melspectrogram_x)
#mfcc
mfcc_input = Reshape(target_shape=(130,2048,1))(inputs)
mfcc_x = Conv1D(128,kernel_size=512, strides=1, padding='valid', activation='relu',name = "mfcc"+"_1")(mfcc_input)
mfcc_x = MaxPooling2D(pool_size=(1,512), strides = (1,220), padding = 'valid',name="mfcc_max")(mfcc_x)
# mfcc_x = Flatten(name="mfcc_flat")(mfcc_x)
mfcc_x = Reshape(target_shape=(130,5*128),name="mfcc_reshape")(mfcc_x)
mfcc_x = Dense(256, activation="relu",name = "mfcc"+"_2")(mfcc_x)
mfcc_x = Dense(128, activation="relu",name = "mfcc"+"_3")(mfcc_x)
mfcc_outputs = Dense(20,activation="linear", name="mfcc"+"_output")(mfcc_x)
concat = Concatenate()([zcr_outputs,chroma_outputs,mfcc_outputs,rms_outputs,melspectrogram_outputs])
feature_output = AveragePooling1D(pool_size=130)(concat)
feature_output = Reshape(target_shape=(162,1))(feature_output)
layer = Conv1D(256, 8, padding='same')(feature_output)  # X_train.shape[1] = No. of Columns
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
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

print("declared model")
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

#Trainable FIN or not
for i in range(38):
    model.layers[i].trainable = True

print("changed model trainability status")


rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)

history=model.fit(training_generator, epochs=50, validation_data=validation_generator, callbacks=[rlrp])

print("Accuracy of our model on test data : " , model.evaluate(test_generator)[1]*100 , "%")

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
plt.savefig("consolidated_loss_curve.png")
model.save("consolidtaed_model")

# predicting on test data.
pred_test = model.predict(test_generator)
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
plt.savefig("consolidated_confusion_matrix.png")

print(classification_report(y_test, y_pred))

