import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import librosa

import sound_utils
from utils import *
from sound_msgs.msg import *
from geometry_msgs.msg import *
from visualization_msgs.msg import *

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
# COLORS = ['b1', 'g2', 'r3', 'c4']
# COLORS = ['b', 'g', 'r', 'k']
# MARKERS = ['o', 'h', 's', 'v']
# MARKERS = ['x', '+', '*', '1']
FREQ = 500
N_FREQS = int( 1 + FRAME_SIZE/2 )
IDX_FREQ = round( N_FREQS * FREQ / (SAMPLE_RATE/2) )
Y_RANGE = [0, 100]
# Y_RANGE = [-80, 10]
AMP_WINDOW_LEN = 20

# Save plots
PLOT_SECS = 20
F_MAX = 2000


class SoundSpec():
    def __init__(self):
        self.plot_spectrogram()
        self.specs_stored = []
        rospy.Subscriber('raw_data', SoundSample, self.mic_data_callback)
        # self.anim = animation.FuncAnimation(self.fig, self.animate, interval=1000)
        plt.show()

    # def animate(self, frame):
    #     rospy.Subscriber('raw_data', SoundSample, self.mic_data_callback)

    def mic_data_callback(self, data):
        for i, mic in enumerate(data.frame_arrays):
            magnitude = []
            for frame in mic.frames:
                amp = np.array([point.amplitude for point in frame.points])
                amp = librosa.amplitude_to_db(amp, ref=HEARING_THRESHOLD)
                # amp = 20 * np.log10(amp/HEARING_THRESHOLD)
                magnitude.append(amp)
            self.specs_stored.append(magnitude)
            stft_size = len(magnitude)
            magnitude = np.array(magnitude).T
            self.stft[i][:, -stft_size:] = magnitude
            self.imgs[i].set_data(self.stft[i])
            self.stft[i][:, 0:-stft_size] = self.stft[i][:, stft_size:]
            self.amps[i].append( np.average(magnitude[IDX_FREQ, :]) )
            # self.amps[i].append( np.average(magnitude) )

        self.amp.clear()
        self.amp.set_ylim([Y_RANGE[0], Y_RANGE[1]])
        len_amps = len(self.amps[0])
        if len_amps < AMP_WINDOW_LEN:
            X = np.arange(0, len_amps)
            for i in range(N_CHANNELS):
                self.amp.plot(X, self.amps[i], COLORS[i])
        else:
            X = np.arange(0, AMP_WINDOW_LEN)
            for i in range(N_CHANNELS):
                self.amp.plot(X, self.amps[i][-AMP_WINDOW_LEN:], COLORS[i])

        self.amp.legend(['mic'+str(i+1) for i in range(N_CHANNELS)], loc='lower right')
        rospy.sleep(TIME_FRAME)


    def run(self):
        rospy.spin()
        self.save_plots()

    
    def plot_spectrogram(self):
        self.fig = plt.figure()
        ax = []
        self.fig.tight_layout(pad=0)
        self.imgs = []
        self.stft = []
        self.amps = []
        f_max_idx = int(FRAME_SIZE / 2) + 1 # number of lines in the spectrogram
        cols_stft = 2 + int( FRAME_LENGTH / HOP_LENGTH ) # number of columns per frame
        cols_plot = 13 # number of columns of plot (figure window length)

        self.amp = self.fig.add_subplot(2, 1, 2)

        for i in range(N_CHANNELS):
            self.stft.append( np.full((f_max_idx, cols_stft*cols_plot), 0) )
            self.amps.append([])
            ax = self.fig.add_subplot(2, N_CHANNELS, i+1)
            self.imgs.append( ax.imshow(self.stft[i], aspect='auto', origin='lower') )
            if i == 0:
                ax.set_ylabel(f'Frequency')
                step_label = 1000 #Hz
                step = step_label*f_max_idx/(SAMPLE_RATE/2)
                yticks = np.arange(0, f_max_idx+step, step)
                yticks = np.around(yticks).astype(int)
                yticks_labels = np.arange(0, int(SAMPLE_RATE/2)+step_label, step_label)
                ax.set_yticks(yticks, yticks_labels)
            else:
                ax.set_yticks([])
            ax.set_title('mic' + str(i+1))
            self.imgs[i].set_clim(0, 100)
            # self.imgs[i].set_clim(-80, 0)
        self.fig.colorbar(self.imgs[i], ax=ax, format='%+2.0f dB')
        self.fig.subplots_adjust(bottom=0.1, left=0.1, top = 0.9, right=0.9, hspace=0.2, wspace=0.05)
        self.fig.show()

    
    def save_plots(self):
        f_max_idx = int(FRAME_SIZE / 2) + 1
        f_max_idx = int( F_MAX*f_max_idx/(SAMPLE_RATE/2) )
        frames_per_sec = 2 + int( SAMPLE_RATE / HOP_LENGTH )
        specs = []
        if len(self.specs_stored) > PLOT_SECS:
            self.specs_stored = self.specs_stored[:N_CHANNELS*PLOT_SECS]
        for i, frame in enumerate(self.specs_stored):
            spec = np.array(frame).T
            spec = spec[:f_max_idx, :]
            if i < N_CHANNELS:
                specs.append(spec)
            else:
                specs[i%N_CHANNELS] = np.append(specs[i%N_CHANNELS], spec, axis=1)

        imgs = []
        fig, axs = plt.subplots(N_CHANNELS, 1)
        if N_CHANNELS==1:
            axs = [axs]
        step_label = 400 #Hz
        yticks_labels = np.arange(0, F_MAX+step_label, step_label)
        # step = step_label*1000*f_max_idx/F_MAX
        # yticks = np.arange(0, f_max_idx+step, step)
        # yticks = np.around(yticks).astype(int)
        yticks = []
        for ylabel in yticks_labels:
            ytick = int( ylabel*f_max_idx/F_MAX )
            yticks.append(ytick)
        # yticks_labels = np.divide(yticks_labels, 100).astype(int)
        for i in range(N_CHANNELS):
            imgs.append( axs[i].imshow(specs[i], aspect='auto', origin='lower') )
            axs[i].set_ylabel('Freq (Hz)')
            axs[i].set_yticks(yticks, yticks_labels)
            imgs[i].set_clim(0, 100)
            # imgs[i].set_clim(-80, 0)
            axs[i].set_title('mic{}'.format(i+1), pad=1, fontsize=10)
            if i<3:
                axs[i].set_xticks([])
        t_max_idx = int( len(self.specs_stored) / N_CHANNELS )
        step_label = int(t_max_idx/10)+1
        step = step_label*frames_per_sec
        xticks = np.arange(0, specs[0].shape[1]+step, step)
        xticks = np.around(xticks).astype(int)
        xticks_labels = np.arange(0, t_max_idx+step_label, step_label)
        axs[i].set_xticks(xticks, xticks_labels)
        axs[i].set_xlabel('Time (s)')
        cax = plt.axes([0.8, 0.07, 0.05, 0.88]) # [left, bottom, width, height]

        # cax = plt.axes([0.8, 0.02, 0.05, 0.93]) # [left, bottom, width, height]
        fig.colorbar(mappable=imgs[i], cax=cax, format='%+2.0f dB')
        fig.subplots_adjust(bottom=0.07, left=0.15, top = 0.95, right=0.75, hspace=0.2, wspace=0.05)
        fig.set_figheight(7)
        fig.set_figwidth(5)
        # fig.set_tight_layout(True)
        fig.savefig('/home/miguel/Spectrograms')
        plt.close(fig)

        # Plot amplitudes
        if len(self.specs_stored) > PLOT_SECS:
            for i in range(N_CHANNELS):
                self.amps[i] = self.amps[i][:PLOT_SECS]
        fig, ax = plt.subplots()
        for i in range(N_CHANNELS):
            X = np.arange(0, len(self.amps[i]))
            X = np.multiply(X, TIME_FRAME)
            ax.plot(X, self.amps[i], COLORS[i])
            print(self.amps[i])
            # ax.plot(X, self.amps[i], linestyle='solid', marker=MARKERS[i], color=COLORS[i], markersize=10)
        ax.legend(['mic'+str(i+1) for i in range(N_CHANNELS)], loc ="lower right")
        # ax.set_ylim([Y_RANGE[0], Y_RANGE[1]])
        ax.set_ylabel('SPL (dB)')
        ax.set_xlabel('Time (s)')
        fig.savefig('/home/miguel/Amps')
        plt.close(fig)    