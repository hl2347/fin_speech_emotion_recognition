import pandas as pd
import numpy as np

import os
import sys
import glob
# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import math

from sklearn.preprocessing import StandardScaler, OneHotEncoder


import keras
from keras import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Activation, Input, MaxPooling2D, Reshape, Concatenate,AveragePooling1D

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
Ravdess = "../input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
Crema = "../input/cremad/AudioWAV/"
Tess = "../input/Tess/Tess/"
Savee = "../input/surrey-audiovisual-expressed-emotion-savee/ALL/"

#Load Dataset
ravdess_directory_list = os.listdir(Ravdess)

file_emotion = []
file_path = []
author_id = []
for dir in ravdess_directory_list:
    # as their are 20 different actors in our previous directory we need to extract files for each actor.
    actor = os.listdir(Ravdess + dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        # third part in each file represents the emotion associated to that file.
        id = 1000 + int(part[-1]) * 10 + (int(part[-1]) % 2)
        author_id.append(str(id))
        file_emotion.append(int(part[2]))
        file_path.append(Ravdess + dir + '/' + file)
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
author_df = pd.DataFrame(author_id, columns=['author_id'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Ravdess_df = pd.concat([emotion_df, path_df, author_df], axis=1)

# changing integers to actual emotions.
Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)

crema_directory_list = os.listdir(Crema)

file_emotion = []
file_path = []
author_id = []

for file in crema_directory_list:
    # storing file paths
    file_path.append(Crema + file)
    # storing file emotions
    part=file.split('_')
    id = 2000 + (int(part[0])-1000)*10 + 2
    author_id.append(str(id))
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
author_df = pd.DataFrame(author_id, columns=['author_id'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Crema_df = pd.concat([emotion_df, path_df,author_df], axis=1)

tess_directory_list = os.listdir(Tess)

file_emotion = []
file_path = []
author_id = []

for dir in tess_directory_list:
    directories = os.listdir(Tess + dir)
    for file in directories:
        part = file.split('.')[0]
        age = part.split('_')[0]
        id = 3000
        if(age == "OAF"):
            id += 10
        author_id.append(str(id))

        part = part.split('_')[2]
        if part=='ps':
            file_emotion.append('surprise')
        else:
            file_emotion.append(part)
        file_path.append(Tess + dir + '/' + file)
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
author_df = pd.DataFrame(author_id, columns=['author_id'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Tess_df = pd.concat([emotion_df, path_df, author_df], axis=1)

savee_directory_list = os.listdir(Savee)

file_emotion = []
file_path = []
author_id = []

for file in savee_directory_list:
    file_path.append(Savee + file)

    id = file.split('_')[0]
    code = 4001
    if id == "JE":
        code += 10
    elif id == "JK":
        code += 20
    elif id == "KL":
        code += 30
    author_id += str(code)


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
author_df = pd.DataFrame(author_id, columns=['author_id'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Savee_df = pd.concat([emotion_df, path_df,author_df], axis=1)

data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis = 0)
data_path_list = [Ravdess_df, Crema_df, Tess_df, Savee_df]
dataset_list = ["ravdess","crema","tess","savee"]
print("loaded data")





emotions = ['angry','calm','disgust','fear','happy','neutral','sad','surprise']
emotion_dict = {'angry':0,'calm':1,'disgust':2,'fear':3,'happy':4,'neutral':5,'sad':6,'surprise':7}
#One-hot-encoder object initialized to "emotion" -> [0,0,0,0,0,0,0,1]
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(emotions).reshape(-1,1)).toarray()

# labels = {}

# for dir in data_path:
#     emotion = dir.split("/")[-1].split("_")[0]
#     labels[dir] = emotion_dict[emotion]

####################Model Construction
trained_model = load_model("../horovod_final/consolidated_trainable")
weights = trained_model.get_weights()

inputs = Input(shape=(1,2048),name="consolidated_input")
# #zcr
zcr_input = Reshape(target_shape=(1,2048,1))(inputs)
zcr_x = Conv1D(16,kernel_size=16, strides=4, padding='valid', activation='relu',name = "zcr_1")(zcr_input)
zcr_x = Reshape(target_shape=(1,509*16),name="zcr_reshape")(zcr_x)
zcr_x = Dense(32,activation="relu", name="zcr_2")(zcr_x)
zcr_x = Dense(32,activation="relu", name="zcr_3")(zcr_x)
zcr_x = Dense(32,activation="relu", name="zcr_4")(zcr_x)
zcr_outputs = Dense(1,activation="relu", name="zcr_output")(zcr_x)

# #chroma
chroma_input = Reshape(target_shape=(1,2048,1))(inputs)
chroma_x = Conv1D(256,kernel_size=512, strides=1, padding='valid', activation='relu',name = "chroma_1")(chroma_input)
chroma_x = MaxPooling2D(pool_size=(1,256), strides = (1,128), padding = 'valid',name="chroma_max")(chroma_x)
chroma_x = Reshape(target_shape=(1,11*256),name="chroma_reshape")(chroma_x)
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
rms_outputs = Reshape(target_shape=(1,1))(rms_outputs)


#melspectrogram
melspectrogram_input = Reshape(target_shape=(1,2048,1))(inputs)
melspectrogram_x = Conv1D(512,kernel_size=512, strides=1, padding='valid', activation='relu',name = "melspectrogram"+"_1")(melspectrogram_input)
melspectrogram_x = MaxPooling2D(pool_size=(1,256), strides = (1,128), padding = 'valid',name="melspectrogram_max")(melspectrogram_x)
melspectrogram_x = Reshape(target_shape=(1,11*512),name="melspectrogram_reshape")(melspectrogram_x)
melspectrogram_outputs = Dense(128,activation="relu", name="melspectrogram"+"_output")(melspectrogram_x)

#mfcc
mfcc_input = Reshape(target_shape=(1,2048,1))(inputs)
mfcc_x = Conv1D(256,kernel_size=512, strides=1, padding='valid', activation='relu',name = "mfcc"+"_1")(mfcc_input)
mfcc_x = MaxPooling2D(pool_size=(1,512), strides = (1,110), padding = 'valid',name="mfcc_max")(mfcc_x)
mfcc_x = Reshape(target_shape=(1,10*256),name="mfcc_reshape")(mfcc_x)
mfcc_x = Dense(256, activation="relu",name = "mfcc"+"_2")(mfcc_x)
mfcc_x = Dense(128, activation="relu",name = "mfcc"+"_3")(mfcc_x)
mfcc_outputs = Dense(20,activation="linear", name="mfcc"+"_output")(mfcc_x)
concat = Concatenate()([zcr_outputs,chroma_outputs,mfcc_outputs,rms_outputs,melspectrogram_outputs])

feature_output = AveragePooling1D(pool_size=1)(concat)
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

#apply weights


model.compile(optimizer = "adam" , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
print("compiled model")

model.set_weights(weights)

feature_model = Model(inputs=inputs, outputs=feature_output)
feature_model.compile(optimizer = "adam" , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

sr = 22050

mappings = pd.read_csv("train_mappings_long.csv",header=None)
mappings.columns = ["file","original"]


SPLIT_PER_DATASET = 25
job_segment = int(sys.argv[1])
dataset_code = int(job_segment/SPLIT_PER_DATASET)
fragment_code = job_segment % SPLIT_PER_DATASET



dataset = data_path_list[dataset_code]
dataset_name = dataset_list[dataset_code]
emotion_count_list = [0 for i in range(8)]
dataset_len = len(dataset)
lines_per_dataset = math.ceil(dataset_len / SPLIT_PER_DATASET)
st = max(0, fragment_code * lines_per_dataset)
ed = min(dataset_len-1, (fragment_code+1)*lines_per_dataset)
dataset = pd.DataFrame(dataset.iloc[st:ed,:])

count = 0

original_path = []
predicted_path = []
for path, emotion, author_id in zip(dataset["Path"],dataset["Emotions"],dataset["author_id"]):
    if path in list(mappings["original"]):
        if emotion:
            emotion_code = emotion_dict[emotion]

            correct = 0

            data, sr = librosa.load(path)

            x = preprocess_data(data)
            x_collective = preprocess_data_collective(data)
            trimmed = adjust_length(data,2.5)

            utterance_pred = trained_model.predict(np.array(x_collective))
            audio_pred = emotions[utterance_pred.argmax()]
            pred_emotion = utterance_pred.argmax()
            # if audio_pred == emotion:
            #     correct = 1

            pred = model.predict(x[0])
            prob = np.array([i[emotion_code] for i in pred])
            features_pred = feature_model.predict(x[0])

            # correctness = []
            frame_index = []
            frames = []
            author_list = []
            features = []
            for i in range(108):
                frames.append(x[0][i][0])
                # correctness.append(correct)
                frame_index.append(i)
                author_list.append(author_id)
                feature = features_pred[i].reshape(1,162)[0]
                features.append(feature)
            if len(frames) > 0:
                df = pd.DataFrame(frames)
                df["frame_index"] = frame_index
                df["utterance_pred"] = pred_emotion
                df["author"] = author_list
                df["prob"] = prob
                for i in range(8):
                    df["probability_"+str(i)] = pred[:,i]
                title = dataset_name+"/"+emotion+"_"+str(fragment_code * lines_per_dataset+count)+"_"+str(pred_emotion)+".csv"
                df.to_csv("frame_probability/"+title, index=False)

                feature_df = pd.DataFrame(features)
                feature_df["frame_index"] = frame_index
                feature_df.to_csv("feature_probability/"+title, index=False)

                original_path.append(path)
                predicted_path.append(title)
                count += 1
            emotion_count_list[emotion_code] += 1

mappings = pd.DataFrame([])
mappings["original"] = original_path
mappings["predicted"] = predicted_path
mappings.to_csv("original_to_frame_predicted_mappings.csv",index=False, header=None, mode="a")