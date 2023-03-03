import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

# to play the audio files
from IPython.display import Audio

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization,Activation
# from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


#Data path
train = pd.read_csv("../baseline_features/train_features.csv",index_col=0)
train = train.sample(frac=1)
train_x = train.iloc[:,:-1].values
train_y = train["labels"].values

val = pd.read_csv("../baseline_features/validation_features.csv",index_col=0)
val = val.sample(frac=1)
val_x = val.iloc[:,:-1].values
val_y = val["labels"].values

test = pd.read_csv("../baseline_features/test_features.csv",index_col=0)
test = test.sample(frac=1)
test_x = test.iloc[:,:-1].values
test_y = test["labels"].values


# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
# print(train_y)
train_y = encoder.fit_transform(np.array(train_y).reshape(-1,1)).toarray()
val_y = encoder.transform(np.array(val_y).reshape(-1,1)).toarray()
test_y = encoder.transform(np.array(test_y).reshape(-1,1)).toarray()

# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
val_x = scaler.transform(val_x)
test_x = scaler.transform(test_x)
# train_x = np.expand_dims(train_x, axis=2)
# val_x = np.expand_dims(val_x, axis=2)
# test_x = np.expand_dims(test_x, axis=2)
print("train: "+ str(len(train_x)))
print("validation: "+ str(len(val_x)))
print("test: "+ str(len(test_x)))
print(train_y[0])
print(len(train_y[0]))

model = Sequential()
model.add(Dense(1024,input_shape=(162,), activation='relu', name="rms_1"))
model.add(Dense(512, activation='relu',name="rms_2"))
model.add(Dense(256, activation='relu',name="rms_3"))
model.add(Dense(128, activation='relu',name="rms_4"))
model.add(Dense(64, activation='relu',name="rms_5"))
model.add(Dense(8, activation='softmax',name="output"))
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
history=model.fit(train_x, train_y, batch_size=64, epochs=50, validation_data=(val_x, val_y), callbacks=[rlrp])

# y_pred = model.predict(test_x)

# print(confusion_matrix(test_y,y_pred))
# print(classification_report(test_y,y_pred))

print("Accuracy of our model on test data : " , model.evaluate(test_x,test_y)[1]*100 , "%")

epochs = [i for i in range(50)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']

fig.set_size_inches(20,6)
ax[0].plot(epochs , train_loss , label = 'Training Loss')
ax[0].plot(epochs , test_loss , label = 'Testing Loss')
ax[0].set_title('Training & Validation Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
ax[1].set_title('Training & Validation Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
# plt.savefig("new_baseline_model_precalculated_loss_curve.png")
# model.save('new_baseline_model_precalculated')

# predicting on test data.
pred_test = model.predict(test_x)
y_pred = encoder.inverse_transform(pred_test)

y_test = encoder.inverse_transform(test_y)

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
# plt.savefig("new_baseline_model_precalculated.png")

print(classification_report(y_test, y_pred))

