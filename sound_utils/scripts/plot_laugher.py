import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
import soundfile as sf
import scipy.signal as sci

matplotlib.rcParams.update({'font.size': 15})

SAMPLE_RATE = 44100
FRAME_SIZE = 2048
HOP_LENGTH = 1024
HEARING_THRESHOLD = 1*10**-12 
DATA_PATH = '/home/miguel/sound_classification/src/datasets/dcase2016_task2_train_dev/plot_test_data'
labelID = ['Clear throat', 'Door slam', 'Drawer', 'Keyboard', 'Keys drop', 'Knock', 'Laughter', 'Page turn', 'Phone', 'Speech']
time_steps = [6, 4, 9, 12, 3, 8, 23, 21, 19, 21]


label_index = 6
# time_step = time_steps[label_index]
time_step = 15

file_path = '/home/miguel/sound_classification/src/MobileRobots/sound_tools/sound_generator/sounds/laughter144.wav'
signal, sr = sf.read(file_path)
f, t, stft = sci.stft(signal, sr, nperseg=FRAME_SIZE, noverlap=HOP_LENGTH)

f_max = 20000
f_max_idx = int(FRAME_SIZE / 2) + 1
f_max_idx = int( f_max*f_max_idx/(SAMPLE_RATE/2) )
frames_per_sec = 2 + int( SAMPLE_RATE / HOP_LENGTH )

min_id = -1
for i, time in enumerate(t):
    if time > 1:
        min_id = i-1
        break
    if time == f_max:
        min_id = i
        break
print(t[min_id])

max_id = -1
for i, time in enumerate(t):
    if time > 2:
        max_id = i-1
        break
    if time == f_max:
        max_id = i
        break
print(t[max_id])
stft = np.abs(stft)[:, min_id:max_id] # cut highest freqs
t = t[min_id:max_id]

max_id = -1
for i, freq in enumerate(f):
    if freq > f_max:
        max_id = i-1
        break
    if freq == f_max:
        max_id = i
        break
stft = np.abs(stft)[:max_id, :] # cut highest freqs
stft_dB = librosa.amplitude_to_db(stft, ref=HEARING_THRESHOLD)

fig, ax = plt.subplots()
img = ax.imshow(stft_dB, aspect='auto', origin='lower')
ax.set_ylabel('Frequency (kHz)')
# ax.set_xlabel('Time (s)')

step_label = 4000
step = step_label*f_max_idx/f_max
yticks = np.arange(0, f_max_idx+step, step)
yticks = np.around(yticks).astype(int)
# yticks_labels = np.arange(0, int(f_max)+step_label, step_label).round(1)
yticks_labels = [0, 4, 8, 12, 16, 20]
ax.set_yticks(yticks, yticks_labels)
# ax.set_yticks([])

# xticks = np.arange(0, len(t))
# t2 = t[::time_step].round(2)
# xticks = xticks[::time_step]
# ax.set_xticks(xticks, t2)
# ax.set_title(labelID[label_index])
ax.set_xticks([])

# cax = plt.axes([0.8, 0.12, 0.05, 0.78])
# fig.colorbar(mappable=img, cax=cax, format='%+2.0f dB')
fig.set_figwidth(3)
# fig.subplots_adjust(bottom=0.12, left=0.12, top = 0.9, right=0.75, hspace=0.2, wspace=0.05)
# fig.subplots_adjust(bottom=0.12, left=0.12, top = 0.88, right=0.88, hspace=0.2, wspace=0.05)
fig.tight_layout()
  
plt.show()
