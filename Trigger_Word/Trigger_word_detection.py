import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *

# x = graph_spectrogram("audio_examples/example_train.wav")
Ty = 1375
Tx = 5511
n_freq = 101
_, data = wavfile.read("audio_examples/example_train.wav")
print("Time steps in audio recording before spectrogram", data[:,0].shape)
# print("Time steps in input after spectrogram", x.shape)
# Load audio segments using pydub
activates, negatives, backgrounds = load_raw_audio()

print("background len: " + str(len(backgrounds[0])))    # Should be 10,000, since it is a 10 sec clip
print("activate[0] len: " + str(len(activates[0])))     # Maybe around 1000, since an "activate" audio clip is usually around 1 sec (but varies a lot)
print("activate[1] len: " + str(len(activates[1])))     # Different "activate" clips can have different lengths

def get_random_time_segment(segment_ms):
    start = np.random.randint(low=0, high=10000-segment_ms)
    end = start + segment_ms - 1
    return (start, end)

def is_overlapping(segment_time, existing_segments):
    for interval in existing_segments:
        if segment_time[0]<=interval[1]  and segment_time[1] >= interval[0]:
            return True
    return False

def insert_audio_clip(background, audio_clip, previous_segments):
    segment_ms = len(audio_clip)
    segment_time = get_random_time_segment(segment_ms)
    count = 0
    while(is_overlapping(segment_time, previous_segments) and count <20):
        segment_time = get_random_time_segment(segment_ms)
        count +=1
    previous_segments.append(segment_time)
    new_backround = background.overlay(audio_clip, position = segment_time[0])
    return new_backround, segment_time

def insert_ones(y, segment_end_ms):
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    y[0, segment_end_y+1: segment_end_y+51] = 1
    return y

def create_training_example(background, activates, negatives, order):
    existing_segments = []
    y = np.zeros((1, Ty))
    number_of_activate = np.random.randint(0, 5)
    randm_indices = np.random.randint(len(activates), size=number_of_activate)
    activate_sample = [activates[i] for i in randm_indices]
    for activate in activate_sample:
        print("yess")
        background, segment_time = insert_audio_clip(background, activate, existing_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end)
    number_of_negative = np.random.randint(0, 3)
    randm_indices = np.random.randint(len(negatives), size=number_of_negative)
    negative_sample = [negatives[i] for i in randm_indices]
    for negative in negatives:
        background, _ = insert_audio_clip(background, negative, existing_segments)
    # print("Ok")
    background = match_target_amplitude(background, -20)
    file_handle = background.export('triger_train_data/train' + str(order) + '.wav', format='wav')
    print("File (train.wav) was saved in your directory!")
    x = graph_spectrogram('train.wav')
    return x, y

#########################################################################################################################
#########################################################################################################################
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Conv1D, GRU, Dense, Activation, Dropout, BatchNormalization, ReLU, LSTM, Masking, TimeDistributed, Reshape, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

def get_model(input_shape):
    inp = Input(shape= input_shape)
    conv = Conv1D(196, kernel_size=15, strides=4)(inp)
    batch0 = BatchNormalization()(conv)
    re0 = Activation('relu')(batch0)
    drop0 = Dropout(0.8)(re0)
    gru1 = GRU(units=128, return_sequences=True)(drop0)
    drop1 = Dropout(0.8)(gru1)
    batch1 = BatchNormalization()(drop1)
    gru2 = GRU(units=128, return_sequences=True)(batch1)
    drop2 = Dropout(0.8)(gru2)
    batch2 = BatchNormalization()(drop2)
    drop2_2 = Dropout(0.8)(batch2)
    den = TimeDistributed(Dense(1, activation='sigmoid'))(drop2_2)

    model = Model(inputs = inp, outputs = den)
    return model

def detect_triggerword(filename):
    plt.subplot(2,1,1)
    x = graph_spectrogram(filename)
    x = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)
    plt.subplot(2, 1, 2)
    plt.plot(prediction[0, :, 0])
    plt.ylabel('probability')
    plt.show()
    return prediction

chime_file = "audio_examples/chime.wav"
def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime_clip = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    consecutive_timestep = 0
    for i in range(Ty):
        consecutive_timestep +=1
        if predictions[0, i, 0] > threshold and consecutive_timestep > 75:
            audio_clip = audio_clip.overlay(chime_clip, position= ((i/Ty) * audio_clip.duration_seconds)*1000)
            consecutive_timestep = 0
    audio_clip.export('result/trigger_word.wav', format='wav')

import pickle

with open('triger_train_data/train_ex.pkl', 'rb') as f:
    X_train = pickle.load(f)

with open('triger_train_data/train_label.pkl', 'rb') as f:
    Y_train = pickle.load(f)

X_train = np.array(X_train).reshape(-1, Tx, n_freq)
Y_train = np.array(Y_train).reshape(-1, Ty, 1)
# model = get_model((Tx, n_freq))



model = load_model('model.h5', compile=False)
# model.compile(loss= 'binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01), metrics=['accuracy'])
# model.fit(X_train, Y_train, epochs=100, batch_size=16)
# model.save('model.h5')
# detect_triggerword('raw_data/backgrounds/1.wav')
# create_training_example(backgrounds[1], activates, negatives, 1)

def preprocess_audio(filename):
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    # Set frame rate to 44100
    segment = segment.set_frame_rate(44100)
    # Export as wav
    segment.export(filename, format='wav')
file_name = 'audio_examples/dover.wav'
preprocess_audio(file_name)

chime_on_activate('audio_examples/dover.wav', predictions=detect_triggerword('audio_examples/dover.wav'), threshold=0.5)
