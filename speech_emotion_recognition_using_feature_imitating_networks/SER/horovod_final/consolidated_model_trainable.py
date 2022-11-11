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

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

############Horovod
hvd.init() # this initiliases horovod
print('Total GPUs available:', hvd.size())
print('Hi I am a GPU and my rank is:', hvd.rank())

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
  tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
################################################

####Configs
batch_size = 64
learning_rate = 0.001

local_bs = int(batch_size/hvd.size())
scaled_lr = learning_rate/2


#####

################################################
##Custom Data generator & preprocessing function

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(162,1), n_channels=1,
                 n_classes=8, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.nvidia=True
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
            data = pd.read_csv(ID).values.reshape(1,-1)[0]
            X += preprocess_data(data) 
            # Store class
            label = self.labels[ID]
            y.append(to_categorical(label, num_classes=self.n_classes))
        X = np.expand_dims(X, axis=2)
        if(self.nvidia):
            print(os.system('nvidia-smi'),flush=True)
            self.nvidia=False
        return np.array(X), np.array(y)

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
            output.append(data[end_index-frame_length:end_index])
        else:
            output.append(data[start_index:start_index+frame_length])
    output.append(data[frame_length*(-1):])
    return np.array(output)

def preprocess_data(data):

    X = []
    res1 = adjust_length(data,2.5)
    res1 = divide_into_frames(res1)
    X.append(res1)
        
    return X


##############Data preparation
#Data path
data_path = []

train_dir = sorted(glob.glob("../new_directory_augmented/train/*.csv"))
val_dir = sorted(glob.glob("../new_directory_augmented/validation/*.csv"))
test_dir = sorted(glob.glob("../new_directory_augmented/test/*.csv"))

data_path  += train_dir
data_path  += val_dir
data_path  += test_dir


emotions = ['angry','calm','disgust','fear','happy','neutral','sad','surprise']
emotion_dict = {'angry':0,'calm':1,'disgust':2,'fear':3,'happy':4,'neutral':5,'sad':6,'surprise':7}
#One-hot-encoder object initialized to "emotion" -> [0,0,0,0,0,0,0,1]
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(emotions).reshape(-1,1)).toarray()
################################


############Data Generator
#Labeling for data generator
labels = {}

sample_rate = 22050

for dir in data_path:
    emotion = dir.split("/")[-1].split("_")[0]
    labels[dir] = emotion_dict[emotion]

img_ids = list(labels.keys())
# random.Random(0).shuffle(img_ids)

train_test_split = int(0.7 * len(img_ids))

train_img_ids = img_ids[:train_test_split]
random.Random(0).shuffle(train_img_ids)

val_test_ids = img_ids[train_test_split:]
val_test_ids_unaugmented = []
#use only unaugmented data
for dir in val_test_ids:
    file_code = int(dir.split("/")[-1].split("_")[-1].split(".")[0])
    if file_code % 3 == 0:
        val_test_ids_unaugmented.append(dir)


n_gpu = hvd.size()
gpu_index = hvd.rank()
sample_size = int(len(train_img_ids)/n_gpu)

l_index = gpu_index*sample_size
r_index = (gpu_index+1)*sample_size if gpu_index+1 < n_gpu else len(train_img_ids)

train_ids = train_img_ids[l_index:r_index]

validation_size = int(len(val_test_ids_unaugmented)/2)
valid_ids = val_test_ids_unaugmented[:validation_size]
test_ids = val_test_ids_unaugmented[validation_size:]

total_train_batches = len(train_ids)//batch_size
total_val_batches = len(valid_ids)//batch_size


params = {'dim': (108,2048),
          'batch_size': local_bs,
          'n_classes': 8,
          'n_channels': 1,
          'shuffle': True}

partition = {'train':train_ids, 'validation':valid_ids,'test':test_ids}

training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)
test_generator = DataGenerator(partition['test'], labels, **params)
# train_dataset = training_generator.unbatch().shard(hvd.size(),hvd.rank()).batch(local_bs).cache()
# val_dataset = validation_generator.unbatch().shard(hvd.size(),hvd.rank()).batch(local_bs).cache()


y_test = []
for p in test_ids:
     y_test.append(emotions[labels[p]])
y_test = encoder.transform(np.array(y_test).reshape(-1,1)).toarray()


####################Model Construction
#Load FINs
zcr_model=load_model("../FIN_topology/zcr/FIN_zcr")
chroma_model=load_model("../FIN_topology/chroma/chroma_5")
rms_model=load_model("../FIN_topology/rms/FIN_rms")
melspectrogram_model=load_model("../FIN_topology/melspectrogram/melspectrogram_1")
mfcc_model=load_model("../FIN_topology/mfcc/FIN_mfcc_original")
print("loaded models")


#weights
zcr_weights = zcr_model.get_weights()
chroma_weights = chroma_model.get_weights()
rms_weights = rms_model.get_weights()
melspectrogram_weights = melspectrogram_model.get_weights()
mfcc_weights = mfcc_model.get_weights()
print("loaded model weights")



opt = tf.keras.optimizers.Adam(learning_rate=scaled_lr)
# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt,backward_passes_per_step=1, average_aggregated_gradients=True)

#Model construction
inputs = Input(shape=(108,2048),name="consolidated_input")
# #zcr
zcr_input = Reshape(target_shape=(108,2048,1))(inputs)
zcr_x = Conv1D(16,kernel_size=16, strides=4, padding='valid', activation='relu',name = "zcr_1")(zcr_input)
zcr_x = Reshape(target_shape=(108,509*16),name="zcr_reshape")(zcr_x)
zcr_x = Dense(32,activation="relu", name="zcr_2")(zcr_x)
zcr_x = Dense(32,activation="relu", name="zcr_3")(zcr_x)
zcr_x = Dense(32,activation="relu", name="zcr_4")(zcr_x)
zcr_outputs = Dense(1,activation="relu", name="zcr_output")(zcr_x)

# zcr_dense = Dense(1024, activation="relu", name = "zcr"+"_1")
# zcr_x = zcr_dense(inputs)
# zcr_x = Dense(192, activation="tanh",name = "zcr"+"_2")(zcr_x)
# zcr_x = Dense(32, activation="tanh",name = "zcr"+"_3")(zcr_x)
# zcr_x = Dense(32, activation="tanh",name = "zcr"+"_4")(zcr_x)
# zcr_x = Dense(32, activation="tanh",name = "zcr"+"_5")(zcr_x)
# zcr_x = Dense(32, activation="tanh",name = "zcr"+"_6")(zcr_x)
# zcr_x = Dense(32, activation="tanh",name = "zcr"+"_7")(zcr_x)
# zcr_outputs = Dense(1, activation="sigmoid",name="zcr"+"_output")(zcr_x)
# zcr_outputs = Reshape(target_shape=(108,1))(zcr_outputs)

# #chroma
chroma_input = Reshape(target_shape=(108,2048,1))(inputs)
chroma_x = Conv1D(256,kernel_size=512, strides=1, padding='valid', activation='relu',name = "chroma_1")(chroma_input)
chroma_x = MaxPooling2D(pool_size=(1,256), strides = (1,128), padding = 'valid',name="chroma_max")(chroma_x)
# melspectrogram_x = Flatten(name="melspectrogram_flat")(melspectrogram_x)
chroma_x = Reshape(target_shape=(108,11*256),name="chroma_reshape")(chroma_x)
chroma_x = Dense(256,activation="relu", name="chroma_2")(chroma_x)
chroma_x = Dense(128,activation="relu", name="chroma_3")(chroma_x)
chroma_x = Dense(64,activation="relu", name="chroma_4")(chroma_x)
chroma_outputs = Dense(12,activation="relu", name="chroma"+"_output")(chroma_x)
# chroma_outputs = Reshape(target_shape=(108,12))(chroma_outputs)

# chroma_dense = Dense(1024, activation="relu", name = "chroma"+"_1")
# chroma_x = chroma_dense(inputs)
# chroma_x = Dense(1024, activation="relu",name = "chroma"+"_2")(chroma_x)
# chroma_x = Dense(512, activation="relu",name = "chroma"+"_3")(chroma_x)
# chroma_x = Dense(256, activation="relu",name = "chroma"+"_4")(chroma_x)
# chroma_x = Dense(64, activation="relu",name = "chroma"+"_5")(chroma_x)
# chroma_x = Dense(32, activation="relu",name = "chroma"+"_6")(chroma_x)
# chroma_x = Dense(32, activation="relu",name = "chroma"+"_7")(chroma_x)
# chroma_outputs = Dense(12,activation="relu", name="chroma"+"_output")(chroma_x)
# chroma_outputs = Reshape(target_shape=(108,12))(chroma_outputs)

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
# melspectrogram_x = Flatten(name="melspectrogram_flat")(melspectrogram_x)
melspectrogram_x = Reshape(target_shape=(108,11*512),name="melspectrogram_reshape")(melspectrogram_x)
melspectrogram_outputs = Dense(128,activation="relu", name="melspectrogram"+"_output")(melspectrogram_x)

#mfcc
mfcc_input = Reshape(target_shape=(108,2048,1))(inputs)
mfcc_x = Conv1D(256,kernel_size=512, strides=1, padding='valid', activation='relu',name = "mfcc"+"_1")(mfcc_input)
mfcc_x = MaxPooling2D(pool_size=(1,512), strides = (1,110), padding = 'valid',name="mfcc_max")(mfcc_x)
# mfcc_x = Flatten(name="mfcc_flat")(mfcc_x)
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
print("set model weights")

#Trainable FIN or not
for i in range(33):
    model.layers[i].trainable = True
print("changed model trainability status")

model.compile(optimizer = opt , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
print("compiled model")

####################Model Training
rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
# callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),hvd.callbacks.MetricAverageCallback(),rlrp]
callbacks = [hvdk.BroadcastGlobalVariablesCallback(0),hvdk.MetricAverageCallback(),rlrp]
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('consolidated_model_untrainable', save_best_only=True))
if hvd.rank() == 0:
    start_t = time.time()
    train_started = time.time()
history=model.fit(training_generator, batch_size = batch_size*4, epochs=70, validation_data=validation_generator, verbose=2 if hvd.rank() == 0 else 0, callbacks=callbacks)
####################################################################################################


#####################Model stats
if hvd.rank() == 0:
    train_ended = time.time()
    start_t = time.time()
    print("train time: ", train_ended - train_started)
    print("Accuracy of our model on test data : " , model.evaluate(test_generator)[1]*100 , "%")

    epochs = [i for i in range(70)]
    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    test_acc = history.history['val_accuracy']
    test_loss = history.history['val_loss']
    hist = pd.DataFrame(history.history)
    hist.to_csv("consolidated_trainable_log.csv",index=False)

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
    plt.savefig("consolidated_trainable_loss_curve.png")
    model.save("consolidated_trainable")


    test_X = []
    test_y = []
    for i, ID in enumerate(test_ids):
        # Store sample
        data = pd.read_csv(ID).values.reshape(1,-1)[0]
        test_X += preprocess_data(data) 
        # Store class
        label = labels[ID]
        test_y.append(to_categorical(label, num_classes=8))
    # test_X = np.expand_dims(test_X, axis=2)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    # predicting on test data.
    y_pred = []
    for x in test_X:
        pred_test = model.predict(np.array([x]))
        y_pred.append(encoder.inverse_transform(pred_test))
    y_pred = np.array(y_pred)

    y_test = encoder.inverse_transform(test_y)

    df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
    df['Predicted Labels'] = y_pred.flatten()
    df['Actual Labels'] = y_test.flatten()
    df.to_csv("conosolidated_model_trainable_confusion_matrix_data.csv",index=False)

    cm = confusion_matrix(df['Actual Labels'], df['Predicted Labels'])
    plt.figure(figsize = (12, 10))
    cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
    sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    plt.title('Confusion Matrix', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)
    plt.savefig("consolidated_trainable_confusion_matrix.png")

    print(classification_report(df['Actual Labels'], df['Predicted Labels']))

    ########################################################################################################################