## Speech Emotion Recognition using Feature Imitating Networks
In this project, we trained Feature Imitating Networks (FINs) to approximate standard audio features, and explored their usage in Speech Emotion Recognition (SER). We utilized four SER datasets to train five FINs to each predict five audio representations (ZCR, RMS, Chroma STFT, Mel-spectrogram, and MFCCs). We then integrated the FINs in end-to-end SER models; pre-training and then tuning FINs for the SER task yielded the highest F1-score (0.63) outperforming the baseline model (0.56) that used pre-computed feature representations.

### Baseline model
https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition/notebook

### Dataset
The Audio Dataset comes from 5 sources
1. Crowd Sourced Emotional Multimodal Actors Dataset (CREMA-D)
2. RAVDESS Emotional speech audio
3. Surrey Audio-Visual Expressed Emotion (SAVEE)
4. Toronto emotional speech set (TESS)
5. Emotional Speech Database (ESD)

The first four datasets are used to train/test the FINs and classification network. The ESD dataset is used to test the generalizability of the FINs.

The time-span of the audio files are from two to five seconds, and the audio data was processed with a sampling rate of 22kHz. 

In total, 12,161 audio files were used in the project.

### Feature Imitating Networks
Feature imitating networks (FINs) are neural networks pre-trained to output a feature representation (as defined by a closed-form equation) from raw signals as inputs.

Similar to the baseline model, we train FINs to output five speech features: Zero-crossing rate (ZCR), Root mean squared (RMS), Chroma short-time fourier transform (Chroma STFT), Mel-frequency cepstrum coefficients (MFCC), and Melspectrogram.

#### Preprocessing
The audio data was processed with a sampling rate of 22kHz. The audio files were augmented in two ways: the first augmentation approach applied gaussian noise to the original data, while the second augmentation approach slightly reduced the speed and increased pitch of the audio signal (by a factor of 0.8 and 0.7 respectively). The audio was then segmented into 2048 samples (93ms) windows with a 512 sample (25\%) shift.

With the 4 datasets downloaded, you can run following codes to pre-process the audio files to be ready for model training.

Under the directory "codes/pre_processing" run,
```
python3 data_path.py
python3 new_directory.py
```

#### Input (Augio segmentation)
Each audio file was split into segments each of length 2,048 (equivalent to 9.3ms) hopped by 512 (25% of the frame length). This setting was chosen to match the default configurations in the librosa methods. 

#### Output
The feature labels were computed using the librosa library.
The output dimensions for each feature is the following:
1. RMS : 1
2. ZCR : 1
3. Chroma : 12
4. MFCC : 20
5. Melspectrogram : 128


#### Model topology
FINs were designed utilizing two neural network structures. The first structure used only fully connected feed-forward layers, while the second structure also included a one-dimensional convolution layer. RMS FIN was only composed of fully connected layers because the approximation was relatively simple, but for the remaining four features, one-dimensional convolution layers were used to take advan- tage of the kernel operations that made the approximation much more efficient.

Typical FIN models look like the following images,
![RMS_FIN](/codes/FIN_final_codes/rms/rms_structure.png)
Model structure of the RMS FIN

![mfcc_FIN](/codes/FIN_final_codes/mfcc/mfcc_structure.png)
Model structure of the MFCC FIN


In order to train FINs, run the following codes under the directory "codes/FIN_final_codes/FEATURE_NAME/"
```
python3 zcr.py
python3 rms.py
python3 chroma.py
python3 mfcc.py
python3 melspectrogram.py
```

#### FIN training results

|   Model  | Output Dim. | Layers | Params |  R²  | R² (ESD) |
|:--------:|:-----------:|:------:|:------:|:----:|:-------------:|
|    RMS   |      1      |   FC   |   66K  |  98% |      97%      |
|    ZCR   |      1      |  Conv. |  263K  |  93% |      92%      |
|  Chroma  |      12     |  Conv. |  900K  |  93% |      92%      |
|   MFCC   |      20     |  Conv. |  822K  |  92% |      54%      |
| Melspec. |     128     |  Conv. |  983K  | 97%* |      97%*     |


*Variance weighted R² was used to evaluate melspectrogram FIN to compensate high variability of values across the indices.

### Ensemble models
![learning_curves](/speech_emotion_recognition_using_feature_imitating_networks/SER/learning_curves.jpg)