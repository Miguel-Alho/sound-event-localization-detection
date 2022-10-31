import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sci
import os
# import pyroomacoustics as pra
from pyroomacoustics import dB, all_combinations
from pyroomacoustics.directivities import cardioid_func
from pyroomacoustics.doa import spher2cart
import librosa

SAMPLE_RATE = 44100
FRAME_SIZE = 2048
HOP_LENGTH = 1024
# DATA_PATH = '/home/miguel/Downloads/python-sounddevice-0.3.15/'
DATA_PATH = '/home/miguel'
ANGLES = [1, 2, 3, 4]
# ANGLES = [0, 1, 2, 3]
HEARING_THRESHOLD = 1*10**-12 #np.max
FREQ = 500
N_FREQS = int( 1 + FRAME_SIZE/2 )
IDX_FREQ = round( N_FREQS * FREQ / (SAMPLE_RATE/2) )


'''Compare amplitudes'''
amp_angle = dict()
for angle in ANGLES:
    amps = []
    for file in os.listdir(DATA_PATH):
        # if file.startswith('delme_rec_gui_mic{}_'.format(angle)):
        if file.startswith('audio_music_mic0{}'.format(angle)):
            print(angle)
            file_path = os.path.join(DATA_PATH, file)
            data, sr = sf.read(file_path)
            clean_data = data[500:-500]
            freq_array, time_array, complex_amplitude_array = sci.stft(clean_data, SAMPLE_RATE, nperseg=FRAME_SIZE, noverlap=HOP_LENGTH)
            amplitude_array = np.abs(complex_amplitude_array)
            # amplitude_array = dB(complex_amplitude_array)
            amplitude_array = librosa.amplitude_to_db(amplitude_array, ref=HEARING_THRESHOLD)

            # idx_500 = find_nearest(freq_array, value=500)
            amp = np.average(amplitude_array[IDX_FREQ, :])
            # amp = np.sum(amplitude_array)
            amps.append(amp)
            print(amp)
    amp_angle[angle] = np.average(amps)
    print('\n')
print(amp_angle)


'''Plot raw signals'''
# fig, axs = plt.subplots(4, 1)
# for i in ANGLES:
#     file_name = 'audio_500Hz_mic0{}.wav'.format(i)
#     file_path = os.path.join(DATA_PATH, file_name)
#     data, sr = sf.read(file_path)
#     X = np.arange(0, len(data))
#     axs[i-1].plot(X, data)
#     print('plotted')
# plt.show()


'''Plot spectrogram'''
# data, sr = sf.read('/home/miguel/audio_500Hz_mic02.wav')
# freq_array, time_array, complex_amplitude_array = sci.stft(data, SAMPLE_RATE, nperseg=FRAME_SIZE, noverlap=HOP_LENGTH)
# amplitude_array = np.abs(complex_amplitude_array)
# amplitude_array = librosa.amplitude_to_db(amplitude_array, ref=HEARING_THRESHOLD)
# fig, ax = plt.subplots(1, 1)
# img = ax.imshow(amplitude_array, aspect='auto', origin='lower')
# fig.colorbar(mappable=img, ax=ax, format='%+2.0f dB')
# plt.show()

