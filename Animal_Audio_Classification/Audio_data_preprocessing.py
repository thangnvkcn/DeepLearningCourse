import random
import numpy as np
import re
import cv2
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


animal_labels = ['Dog', 'Rooster', 'Pig', 'Cow', 'Frog', 'Cat', 'Hen', 'Insect', 'Sheep', 'Crow']

def get_random_time_segment(len, segment_ms):
    start = np.random.randint(low = 0, high = len-segment_ms)
    end = start + segment_ms - 1
    return (start, end)

def is_overlapping(segment_time, existing_segment):
    for interval in existing_segment:
        if segment_time[0] <= interval[1] and segment_time[1]>= interval[0]:
            return True
    return False

def load_data(path):
    audio_data, label, names = [], [], []
    count = 0

    for subdir in os.listdir(path):
        # print(count)
        for filename in os.listdir(path+'/'+subdir):
            data, sr = librosa.load(path+'/'+subdir+'/'+filename, sr=44100)
            # print(data.shape, " ", sr)
            audio_data.append(data)
            label.append(count)
            names.append(animal_labels[count])
        count+=1

    return audio_data, label, names

def audio_augmentation(audios, labels):
    size_of_audio_data = len(audios)
    for i in range(size_of_audio_data):
        audios.append(audios[i] + 0.005*np.random.randn(audios[0].shape[-1]))
        labels.append(labels[i])
    return audios, labels

def convert_to_spectrogram(audios):
    size_of_audio_data = len(audios)
    audio_spec = []
    for i in range(size_of_audio_data):
        audio_spec.append(librosa.feature.melspectrogram(audios[i], sr=44100))
    del audios
    return audio_spec

from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.utils import  to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

def get_model(input_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return  model


audio_data, label, names = load_data('ESC-50/audio')
number_of_training_example = len(audio_data)
# random_index = np.random.randint(0, number_of_training_example)
# plt.figure(figsize=(20, 5))
# plt.subplot(121)
#
# audio = audio_data[random_index]
# spectrogram = librosa.feature.melspectrogram(audio)
# plt.title("Spectrogram")
# librosa.display.specshow(spectrogram, y_axis='mel', x_axis='time')
# sample_rate = 44100
# plt.subplot(122)
# plt.title("Wave")
# librosa.display.waveplot(audio, sr = sample_rate)
# plt.ylabel("Amplitude")
# print(random_index)
# print(names[random_index])
# plt.show()
# print(number_of_training_example)
audio_augmentation(audio_data, label)
# print(len(audio_data))
audio_data = convert_to_spectrogram(audio_data)
length = len(audio_data)
spec_h, spec_w = audio_data[0].shape
audio_data = np.reshape(audio_data, (length, spec_h, spec_w, 1))

label = to_categorical(label, num_classes=10)
trainX, testX, trainY, testY = train_test_split(audio_data, label, test_size=0.1, shuffle=True, random_state=5)
plt.imshow(trainX[0].reshape(spec_h, spec_w))
# plt.show()
# print(trainX.shape)

model = get_model((spec_h, spec_w, 1))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early = EarlyStopping(monitor='val_loss', mode='min', patience=8)
point = ModelCheckpoint(monitor='val_loss', verbose=0, save_best_only=True, filepath='model.h5')
history=model.fit(trainX, trainY, epochs=20, batch_size=8, verbose=1, validation_split=0.1, callbacks=[early, point] )