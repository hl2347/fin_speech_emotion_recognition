import pandas as pd
import numpy as np

import math

import glob

import librosa
import librosa.display
import sys


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
train_path = sorted(glob.glob("../new_directory/train/*.csv"))
validation_path = sorted(glob.glob("../new_directory/validation/*.csv"))
test_path = sorted(glob.glob("../new_directory/test/*.csv"))
paths = train_path + validation_path + test_path

job_code = int(sys.argv[1])


#Data preprocessing
basedir = "chroma/"
for i, path in enumerate(paths):
    if i % 100 == job_code:
        filename = "/".join(path.split("/")[-2:])
        data = pd.read_csv(path).values.reshape(1,-1)[0]
        frames = divide_into_frames(data)
        chroma_list = []
        for frame in frames:
            stft = np.abs(librosa.stft(frame,n_fft=2048, hop_length=512,center=False))
            chroma_list.append(librosa.feature.chroma_stft(S=stft).reshape(-1,))
        chroma_list = np.array(chroma_list)
        chroma_df = pd.DataFrame(chroma_list)
        chroma_df.to_csv(basedir+filename,header=None,index=None)

