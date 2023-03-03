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
train_x = train.iloc[:,1:-1].values
train_y = train["labels"].values

val = pd.read_csv("../baseline_features/validation_features.csv",index_col=0)
val = val.sample(frac=1)
val_x = val.iloc[:,1:-1].values
val_y = val["labels"].values

test = pd.read_csv("../baseline_features/test_features.csv",index_col=0)
test = test.sample(frac=1)
test_x = test.iloc[:,1:-1].values
test_y = test["labels"].values


# As this is a multiclass classification problem onehotencoding our Y.
# encoder = OneHotEncoder()
# print(train_y)
# train_y = encoder.fit_transform(np.array(train_y).reshape(-1,1)).toarray()
# val_y = encoder.transform(np.array(val_y).reshape(-1,1)).toarray()
# test_y = encoder.transform(np.array(test_y).reshape(-1,1)).toarray()

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

svclassifier = SVC()
svclassifier.fit(train_x, train_y)

y_pred = svclassifier.predict(test_x)

print(confusion_matrix(test_y,y_pred))
print(classification_report(test_y,y_pred))