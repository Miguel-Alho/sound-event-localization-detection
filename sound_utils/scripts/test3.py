from math import log10
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sci
import librosa
import sounddevice as sd

DURATION = 4
SAMPLE_RATE = 44100
HEARING_THRESHOLD = 1*10**-12
FREQ = 200
N_FRAMES = int((SAMPLE_RATE/FREQ)/4)

sd.default.samplerate = SAMPLE_RATE
sd.default.channels = 1
print('start recording')
rec = sd.rec(int(DURATION * SAMPLE_RATE), blocking=True)
# Take the whole signal and compute the maximum
max_amp = np.argmax(rec)
print(max_amp)
print(rec[max_amp])
amp = np.average(rec[max_amp-N_FRAMES : max_amp+N_FRAMES])
# Pass it to dB
amp_dB = 20*log10(amp/HEARING_THRESHOLD)
print(amp_dB)

fig, axs = plt.subplots(1, 1)
X = np.arange(0, len(rec))
axs.plot(X, rec)
plt.show()

