from math import log10
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sci
import os
# import pyroomacoustics as pra
from pyroomacoustics import dB
from pyroomacoustics.directivities import cardioid_func
from pyroomacoustics.doa import spher2cart
import librosa
import sounddevice as sd

from tkinter import *
# from tkinter import filedialog as fd

STFT = False  # if STFT == True, use average of amplitudes for a continuous signal's freq; if False, use the highest peak of an impulse raw signal
DURATION = 4
SAMPLE_RATE = 44100
FRAME_SIZE = 2048
HOP_LENGTH = 1024
HEARING_THRESHOLD = 1*10**-12
FREQ = 500
N_FREQS = int( 1 + FRAME_SIZE/2 )
IDX_FREQ = round( N_FREQS * FREQ / (SAMPLE_RATE/2) )


def change():
    print('change az')
    # TODO: Open a popup window showing all angles and amplitudes, letting the user choose the one he wants to test again

def compute_with_stft(rec):
    print('stft')
    freq_array, time_array, complex_amplitude_array = sci.stft(rec, SAMPLE_RATE, nperseg=FRAME_SIZE, noverlap=HOP_LENGTH)
    amplitude_array = np.abs(complex_amplitude_array)
    amplitude_array = librosa.amplitude_to_db(amplitude_array, ref=HEARING_THRESHOLD)
    return np.average(amplitude_array[IDX_FREQ, :])

def compute_with_raw(rec):
    print('raw')
    max_amp = np.max(abs(rec))
    amp_dB = 20*log10(max_amp/HEARING_THRESHOLD)
    # If this values do not make sense, compare with InputStream
    return amp_dB

def start():
    print('start')
    global var_deg
    global az_index
    # Record input audio
    rec = sd.rec(int(DURATION * SAMPLE_RATE), blocking=True)
    rec = rec[int(0.5*SAMPLE_RATE):]

    X = np.arange(0, len(rec))
    plt.plot(X, rec)
    plt.show()

    if STFT: amp = compute_with_stft(rec)
    else: amp = compute_with_raw(rec)
    amps[az_index] = round(amp, 2)
    var_prev_deg.set('azimuth: {}º'.format(azimuth[az_index]))
    var_prev_amp.set('amp: {} dB'.format(amps[az_index]))
    az_index += 1
    var_deg.set('azimuth: {}º'.format(azimuth[az_index]))
    var_amp.set('amp: {} dB'.format(amps[az_index]))
    

def compute_responses():
    window = Tk()
    window.title("Calibrate mic")
    window.geometry('')

    global var_deg, var_amp, var_prev_deg, var_prev_amp
    global az_index
    az_index = 0

    label_prev = Label(window, text='PREVIOUS:   ')
    label_prev.grid(column=0, row=0)   

    var_prev_deg = StringVar()
    var_prev_deg.set('azimuth: 0º')
    label_prev_deg = Label(window, textvariable=var_prev_deg)
    label_prev_deg.grid(column=1, row=0)

    var_prev_amp = StringVar()
    var_prev_amp.set('amp: 0 dB')
    label_prev_amp = Label(window, textvariable=var_prev_amp)
    label_prev_amp.grid(column=2, row=0)

    label_curr = Label(window, text='CURRENT:   ')
    label_curr.grid(column=0, row=1)   

    var_deg = StringVar()
    var_deg.set('azimuth: {}º'.format(azimuth[az_index]))
    label_deg = Label(window, textvariable=var_deg)
    label_deg.grid(column=1, row=1)

    var_amp = StringVar()
    var_amp.set('amp: {} dB'.format(0))
    label_amp = Label(window, textvariable=var_amp)
    label_amp.grid(column=2, row=1)

    btn1 = Button(window, text='Start', command=start)
    btn1.grid(column=0, row=2)
    btn2 = Button(window, text='Change azimuth', command=change)
    btn2.grid(column=1, row=2)

    window.mainloop() 


sd.default.samplerate = SAMPLE_RATE
sd.default.channels = 1
azimuth = np.arange(0, 360, 20)
amps = np.zeros(len(azimuth))
# lower_gain = -40

# """ 2D """
# # get cartesian coordinates
# cart = spher2cart(azimuth=np.radians(azimuth))

# # compute response
compute_responses()
print(amps)

# # plot
# plt.figure()
# plt.polar(np.radians(azimuth), resp)
# plt.ylim([lower_gain, 0])
# ax = plt.gca()
# ax.yaxis.set_ticks(np.arange(start=lower_gain, stop=5, step=10))
# plt.tight_layout()

