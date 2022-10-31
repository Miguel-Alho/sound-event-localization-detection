#!/usr/bin/env python3

#Importing the required libraries
import os
import sys
import time
import pandas as pd
import numpy as np
from random import shuffle
from tqdm import tqdm
import soundfile as sf
import scipy.signal as sci
import librosa
import librosa.display
import random
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
import matplotlib.pyplot as plt

import sound_utils
from utils import *

# SAMPLE_RATE = 44100
# FRAME_SIZE = 2048
# HOP_LENGTH = 1024
# HEARING_THRESHOLD = 1*10**-12 #np.max
# TIME_FRAME = 1
# FRAME_LENGTH = int(TIME_FRAME*SAMPLE_RATE)
# MIN_AMP_RATIO_THRESHOLD = 0 # it means all amplitudes below 5% of max stft aplitude will be discarded
# FREQ_RANGE = [0, 4000]

'''Setting up the env'''
# labelID = ['clearthroat', 'cough', 'doorslam', 'drawer', 'keyboard', 'keysDrop', 'knock', 'laughter', 'pageturn', 'phone', 'speech']
labelID = ['clearthroat', 'doorslam', 'drawer', 'keyboard', 'keysDrop', 'knock', 'laughter', 'pageturn', 'phone', 'speech']
SAMPLES_PER_CLASS = 20
DATASET_SIZE = len(labelID) * SAMPLES_PER_CLASS


def parse_metadata(data_path):
    metadata = []
    list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(data_path, x)),
                os.listdir(data_path) ) )
    for file in list_of_files:
        label = file[:-7]
        if not label in labelID:
            continue
        classID = labelID.index(label)
        metadata.append((file, classID))
    return metadata


def save_spectrogram(stft, file_name):
        # fig = plt.figure(figsize=(15, 17))
        # librosa.display.specshow(stft_dB, x_axis='time', y_axis="log")
        # plt.colorbar()

        f_max = 4000
        f_max_idx = int(FRAME_SIZE / 2) + 1
        f_max_idx = int( f_max*f_max_idx/(SAMPLE_RATE/2) )
        frames_per_sec = 2 + int( SAMPLE_RATE / HOP_LENGTH )

        fig, ax = plt.subplots()
        img = ax.imshow(stft, aspect='auto', origin='lower')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')

        step_label = 500
        step = step_label*f_max_idx/f_max
        yticks = np.arange(0, f_max_idx+step, step)
        yticks = np.around(yticks).astype(int)
        yticks_labels = np.arange(0, int(f_max)+step_label, step_label).round(1)
        ax.set_yticks(yticks, yticks_labels)
        # ax.set_yticks([])

        time_frames = stft.shape[1]
        step = time_frames / 5
        xticks = np.arange(0, time_frames+step, step)
        xticks = np.around(xticks).astype(int)
        xticks = [0, 9, 18, 27, 36, 44]
        xticks_labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_xticks(xticks, xticks_labels)

        cax = plt.axes([0.8, 0.1, 0.05, 0.8])
        fig.colorbar(mappable=img, cax=cax, format='%+2.0f dB')
        fig.subplots_adjust(bottom=0.1, left=0.12, top = 0.9, right=0.75, hspace=0.2, wspace=0.05)

        fig.savefig('/home/miguel/Spectrograms/{}'.format(file_name))
        plt.close(fig)


def add_noise(data):
    noise = np.random.randn(len(data))
    data_noise = data + 0.005 * noise
    return data_noise

def shift(data):
    shift_time = 0.2
    n_shifts = int(TIME_FRAME/shift_time)
    n_samples = int(SAMPLE_RATE*shift_time)
    signals = []
    for _ in range(n_shifts):
        signals.append( data )
        data = np.roll(data, n_samples)
    if len(data) > FRAME_LENGTH:
        signals = [sig[:FRAME_LENGTH] for sig in signals]
    return signals

def stretch(data, rate=1):
    input_length = 16000
    data = librosa.effects.time_stretch(data, rate)
    if len(data) > input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data


def get_spectrogram(signal, sr):
    f, t, stft = sci.stft(signal, sr, nperseg=FRAME_SIZE, noverlap=HOP_LENGTH)
    min_id = -1
    max_id = -1
    for i, freq in enumerate(f):
        if min_id == -1:
            if freq >= FREQ_RANGE[0]:
                min_id = i
        else:
            if freq > FREQ_RANGE[1]:
                max_id = i-1
                break
            if freq == FREQ_RANGE[1]:
                max_id = i
                break
    stft = np.abs(stft)[min_id:max_id+1, :] # cut lowest and highest freqs
    stft = np.where( stft < MIN_AMP_RATIO_THRESHOLD*np.max(stft), 0, stft ) # remove amplitudes with less energy
    stft_dB = librosa.amplitude_to_db(stft, ref=HEARING_THRESHOLD)
    return stft_dB


'''Creating the training and testing data'''
def create_data_features(data_path, augment):
    metadata = parse_metadata(data_path)
    data_features = []
    for row in tqdm(metadata):
        file, classID = row
        label = np.zeros(len(labelID), dtype=int)
        label[classID] = 1
        audio_path = os.path.join(data_path, file)
        signal, sr = sf.read(audio_path)
        if len(signal.shape) == 2 or signal[0].shape == 2:
            print('\nMore than one channel found...')
        if sr != SAMPLE_RATE:
            print('\nSample rate different from {}!\n'.format(sr))
        signal_length = signal.shape[0]
        if signal_length < FRAME_LENGTH:
            difference = FRAME_LENGTH - signal_length
            zero_pad = np.zeros(difference)
            signal = np.append(signal, zero_pad)
        else:
            if not augment:
                signal = signal[:FRAME_LENGTH]

        if augment:
            signals = []
            signal_noise = add_noise(signal)
            signals.extend( shift(signal) )
            signals.extend( shift(signal_noise) )
            # signals.extend( stretch(signal) )
            # signals.extend( stretch(signal_noise) )
            # print(len(signals))
            for i, signal in enumerate(signals):
                stft_dB = get_spectrogram(signal, sr)
                data_features.append((stft_dB, label))
                # save_spectrogram(stft_dB, file[:-4]+str(i))
        else:
            if signal.shape[0] != FRAME_LENGTH:
                print('\nERROR: A signal has a length of {} samples...'.format(signal.shape[0]))
            stft_dB = get_spectrogram(signal, sr)
            data_features.append((stft_dB, label))
            # save_spectrogram(stft_dB, file[:-4])
    return data_features

# start = time.time()
# create_data_features('/home/miguel/sound_classification/src/MobileRobots/thesis_ws/src/my_odas_pkg/src_SED7/spectrograms')
# end = time.time()
# print('Running time:')
# print(end - start)