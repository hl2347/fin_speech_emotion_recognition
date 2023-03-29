import pandas as pd
import numpy as np

import sys
import glob
# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage
import math


from scipy.signal import find_peaks
from sklearn.cluster import KMeans

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

def create_spectrogram(data, sr, title):
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(1.5, 1.5))
    plt.axis('off')
#     librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')   
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.savefig(title,bbox_inches='tight', pad_inches=0)
    plt.clf()


def getImage(path, zoom=0.1):
    return OffsetImage(plt.imread(path), zoom=zoom)

def compute_sse(features, centers, labels):
    sse = 0
    for i, feature in enumerate(features):
        label = labels[i]
        center = centers[label]
        sse += sum(np.square(feature-center))
    return sse

def compute_elbow(sse_list):
    dist_list = []
    for i in range(len(sse_list)):
        dist_list.append(np.square(i+2) + np.square(sse_list[i]))
    return np.argmin(dist_list) + 2

dataset_names =["ravdess","savee","crema","tess"]

job_code = int(sys.argv[1])
dataset_name = "ravdess"
if job_code > 7 and job_code < 15:
    dataset_name = "savee"
elif job_code > 14 and job_code < 21:
    dataset_name = "crema"
elif job_code > 20:
    dataset_name = "tess"


frame_data_paths = sorted(glob.glob("frame_probability/"+dataset_name+"/*.csv"))
feature_data_paths = sorted(glob.glob("feature_probability/"+dataset_name+"/*.csv"))


emotion_range_dict = {"ravdess":['angry','calm','disgust','fear','happy','neutral','sad','surprise'],
    "savee":['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
    "crema":['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'],
    "tess":['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']}


emotions = ['angry','calm','disgust','fear','happy','neutral','sad','surprise']
emotion_range = emotion_range_dict[dataset_name]

dataset_start_index = {"ravdess": 0 , "savee": 8, "crema":15, "tess":21}
emotion = emotion_range[job_code - dataset_start_index[dataset_name]]

#generate randomly sampled index
path_df = pd.DataFrame([])
temp_emotion_list = []
temp_path_list = []
temp_feature_path_list = []
category = []
for path, feature_path in zip(frame_data_paths, feature_data_paths):
    file_emotion = path.split("/")[-1].split("_")[0]
    pred_emotion_code = int(path.split("/")[-1].split("_")[-1].split(".")[0])
    pred_emotion = emotions[pred_emotion_code]
    if file_emotion == emotion:
        temp_emotion_list.append(emotion)
        temp_path_list.append(path)
        temp_feature_path_list.append(feature_path)
        if pred_emotion == file_emotion:
            category.append("TP")
        else:
            category.append("FN")
    else:
        temp_emotion_list.append(emotion)
        temp_path_list.append(path)
        temp_feature_path_list.append(feature_path)
        if pred_emotion == emotion:
            category.append("FP")
        else:
            category.append("TN")


path_df["emotion"]= temp_emotion_list
path_df["path"] = temp_path_list
path_df["feature_path"] = temp_feature_path_list
path_df["category"] = category

# random_index_dict = {}

sample_rate = {"ravdess": 0.5 , "savee": 0.5, "crema":0.2, "tess":0.4}
# random_index = list(path_df.sample(frac=sample_rate[dataset_name], random_state = 1).index)
random_index = list(path_df.sample(frac=1.0, random_state = 1).index)


#peak/dip visualization will be plotted for each emotion
X = pd.DataFrame([])
peak_threshold = 0.98
# iterate through all the frame probability files
for i in range(len(path_df)):
    frame_data_path = path_df.iloc[i,1]
    feature_data_path = path_df.iloc[i,2]
    current_file_emotion = path_df.iloc[i,0]
    category = path_df["category"][i]
    current_emotion_code = emotions.index(current_file_emotion)

    # only consider the corresponding emotion
    if i in random_index:
        feature_data = pd.read_csv(feature_data_path)
        feature_data["path"] = frame_data_path
        feature_data["category"] = category
        frame_data = pd.read_csv(frame_data_path)

        
        current_prob = frame_data["probability_"+str(current_emotion_code)]
        peak_index , _ = find_peaks(current_prob, height=0.90)

        frame_data["dip_prob"] = 1-current_prob
        dip_index , _ = find_peaks(frame_data["dip_prob"], height = 0.98)

        stats = pd.DataFrame([[len(peak_index),len(dip_index),emotion,dataset_name]])
        # stats.to_csv("peak_selection_stats.csv",index=False,header=False, mode="a")

        if len(peak_index) < len(dip_index):
            if len(peak_index) > 0:
                dip_index = dip_index[:len(peak_index)-1]
            else:
                dip_index = list(range(1,1))



        peak_frames = pd.DataFrame(feature_data.iloc[peak_index,:])
        peak_frames["peak"] = 1
        dip_frames = pd.DataFrame(feature_data.iloc[dip_index,:])
        dip_frames["peak"] = 0
        meaningful_frames = pd.concat([peak_frames,dip_frames])
        X = pd.concat([X,meaningful_frames])

print("peak: ", len(X.loc[X["peak"] == 1]))
print("dip: ", len(X.loc[X["peak"] == 0]))



#draw spectrogram
# spec_path = []
# for path, frame_index, peak in zip(X["path"], X["frame_index"], X["peak"]):
#     filename = path.split("/")[1].split(".")[0]
#     frame_path = dataset_name+"_frame_probability/"+ filename + ".csv"
#     data = pd.read_csv(frame_path)
#     frame_data = np.array(pd.DataFrame(data.loc[data["frame_index"] == frame_index]).iloc[0,:2048])
#     title = "new_big_spectrograms/"+dataset_name + "/" + filename + "_"+str(peak)+"_" + str(frame_index) + ".png"
#     spec_path.append(title)
#     create_spectrogram(frame_data,22050,title)

# #Add border
# for file in sorted(glob.glob("new_big_spectrograms/*/*.png")):
#     spec_emotion = file.split("/")[-1].split("_")[0]
#     peak = int(file.split("/")[-1].split("_")[-2])
#     if spec_emotion == emotion:
#         c = [0,0,0]
#         if peak == 1: # peak
#             c = [255,255,0]
#         else:
#             c = [100,100,255]
#         img = cv2.imread(file)

#         # border widths; I set them all to 150
#         top, bottom, left, right = [5] * 4

#         img_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=c)
#         cv2.imwrite(file, img_with_border)

# X["spec_path"]= spec_path

#tsne calculation with X
# x,y, X,Y, file_index,peak,path, spec_path
feature_data = np.array(X.iloc[:,:162])
sse_list = []
for i in range(2,min(500,len(feature_data))):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(feature_data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    sse = compute_sse(feature_data, centers, labels)
    sse_list.append(sse)

fig, ax = plt.subplots()
title = dataset_name + "_" + emotion
plt.plot(list(range(2,min(500, len(feature_data)))), sse_list)
plt.title(title)
plt.xlabel("number of clusters")
plt.ylabel('sse')
fig.savefig("k_means_elbow_plot/"+ title+".png")
plt.clf()

fig, ax = plt.subplots()
title = dataset_name + "_" + emotion + "_elbow"
end_index = min(100, len(feature_data))
plt.plot(list(range(2,end_index)), sse_list[:end_index-2])
plt.title(title)
plt.xlabel("number of clusters")
plt.ylabel('sse')
fig.savefig("k_means_elbow_plot/"+ title+".png")
plt.clf()

# elbow = compute_elbow(sse_list)
# elbow_df = pd.DataFrame([[job_code, dataset_name, emotion, elbow]])
# elbow_df.to_csv("elbow_list.csv", index=False, header=False, mode="a")
