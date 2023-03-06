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

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# to play the audio files
from IPython.display import Audio

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Activation
# from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


model_number = int(sys.argv[1])
model_code = model_number
binary = []
for i in range(5):
    binary.append(model_code % 2)
    model_code = model_code // 2

#Data path
train_data_path = sorted(glob.glob("ablation_features/train/*.csv"))
val_data_path = sorted(glob.glob("ablation_features/validation/*.csv"))
test_data_path = sorted(glob.glob("ablation_features/test/*.csv"))
data_path = train_data_path + val_data_path + test_data_path

features = []
emotions = []
data_types = []
for path in data_path:
    emotion = path.split("/")[-1].split("_")[0]
    feature = pd.read_csv(path).values.reshape(-1,)
    features.append(feature)
    emotions.append(emotion)
    data_type = path.split("/")[1]
    if data_type == "train":
        data_types.append(0)
    elif data_type == "validation":
        data_types.append(1)
    else:
        data_types.append(2)

features = pd.DataFrame(features)

#FIN -> 0, True -> 1
#ZCR, Chroma, MFCC, RMS, Mel order
#example: FIN_zcr, FIN_chroma, true_mfcc, true_rms, FIN_mel -> 00110 -> 6

FIN_zcr = pd.DataFrame(features.iloc[:,0])
FIN_chroma = pd.DataFrame(features.iloc[:,1:13])
FIN_mfcc = pd.DataFrame(features.iloc[:,13:33])
FIN_rms = pd.DataFrame(features.iloc[:,33])
FIN_mel = pd.DataFrame(features.iloc[:,34:162])

true_zcr = pd.DataFrame(features.iloc[:,162])
true_chroma = pd.DataFrame(features.iloc[:,163:175])
true_mfcc = pd.DataFrame(features.iloc[:,175:195])
true_rms = pd.DataFrame(features.iloc[:,195])
true_mel = pd.DataFrame(features.iloc[:,196:324])

zcr = [FIN_zcr, true_zcr]
chroma = [FIN_chroma, true_chroma]
mfcc = [FIN_mfcc, true_mfcc]
rms = [FIN_rms, true_rms]
mel = [FIN_mel, true_mel]

#ex) 30 --> [0,1,1,1,1]
X = pd.concat([zcr[binary[0]],chroma[binary[1]],mfcc[binary[2]],rms[binary[3]],mel[binary[4]]], axis=1)
X["types"] = data_types
Y = emotions


# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
Y_df = pd.DataFrame(Y)
Y_df["types"] = data_types

# splitting data

x_train = pd.DataFrame(X.loc[X["types"] == 0]).iloc[:,:162]
x_val = pd.DataFrame(X.loc[X["types"] == 1]).iloc[:,:162]
x_test = pd.DataFrame(X.loc[X["types"] == 2]).iloc[:,:162]

y_train = pd.DataFrame(Y_df.loc[Y_df["types"] == 0]).values[:,:-1]
y_val = pd.DataFrame(Y_df.loc[Y_df["types"] == 1]).values[:,:-1]
y_test = pd.DataFrame(Y_df.loc[Y_df["types"] == 2]).values[:,:-1]

# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
x_train = np.expand_dims(x_train, axis=2)
x_val = np.expand_dims(x_val, axis=2)
x_test = np.expand_dims(x_test, axis=2)
print("train: "+ str(len(x_train)))
print("test: "+ str(len(x_test)))
print("Before training")

model = Sequential()
model.add(Conv1D(256, 8, padding='same', input_shape=(x_train.shape[1],1)))  # X_train.shape[1] = No. of Columns
model.add(Activation('relu'))
model.add(Conv1D(256, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(y_train.shape[1])) # Target class number
model.add(Activation('softmax'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
history=model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_val, y_val), callbacks=[rlrp])

print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")

epochs = [i for i in range(100)]
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
plt.savefig("learning_curves/loss_curve_"+str(model_number)+".png")
plt.clf()
model.save("models/"+"model_"+str(model_number))

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
plt.savefig("confusion_matrix/confusion_matrix_"+str(model_number)+".png")
plt.clf()

print(classification_report(y_test, y_pred))

