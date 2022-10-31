import os
import numpy as np
# import rospkg
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.signal as sci
import librosa


TIME_FRAME = 0.1
HEARING_THRESHOLD = 1*10**-12
FRAME_SIZE = 2048
HOP_LENGTH = 1024
SAMPLE_RATE = 44100

FREQ = 500
N_FREQS = int( 1 + FRAME_SIZE/2 )
IDX_FREQ = round( N_FREQS * FREQ / (SAMPLE_RATE/2) )


def save_plot(amps, path):
    fig, ax = plt.subplots()
    print(len(amps))
    for mic_data in amps:
        tmp_freq_array, tmp_time_array, tmp_complex_amplitude_array = sci.stft(mic_data, SAMPLE_RATE, nperseg=FRAME_SIZE, noverlap=HOP_LENGTH)
        tmp_amplitude_array = np.abs(tmp_complex_amplitude_array)
        amps = librosa.amplitude_to_db(tmp_amplitude_array, ref=HEARING_THRESHOLD)

        magnitudes = []
        time = TIME_FRAME
        init_index = 0
        for i, t in enumerate(tmp_time_array):
            if t >= time:
                time += TIME_FRAME
                magnitudes.append( np.average(amps[IDX_FREQ, init_index:(i-1)]) )
                init_index = i
        
        X = np.arange(0, len(magnitudes))
        # X = np.multiply(X, 1/SAMPLE_RATE)
        X = np.multiply(X, TIME_FRAME)
        ax.plot(X, magnitudes)

    ax.legend(['mic'+str(i+1) for i in range(len(amps))], loc ="lower right")
    # ax.set_ylim([Y_RANGE[0], Y_RANGE[1]])
    ax.set_ylabel('SPL (dB)')
    ax.set_xlabel('Time (s)')
    fig.savefig(path)
    # print('Saved: ' + path)
    plt.close(fig)    


if __name__=="__main__":
    # rp = rospkg.RosPack()
    # data_path = rp.get_path('sound_saver') + '/' + 'sounds/living_room/2D_4mics_80cm'
    # data_path = os.getcwd()
    data_path = os.path.realpath(__file__)
    data_path = os.path.dirname(data_path)
    data_path = os.path.dirname(data_path)
    data_path = os.path.dirname(data_path)
    data_path = os.path.join(data_path, 'sound_saver/sounds/garden/2D_4mics_80cm/1_src')
    # data_path = os.path.join(data_path, 'sound_saver/sounds/living_room_B/2D_4mics_80cm')
    print(data_path)

    # for n_src, n_src_folder in enumerate(os.listdir(data_path)):
    #     n_src_path = os.path.join(data_path, n_src_folder)
    #     for test in os.listdir(n_src_path):
    for test in os.listdir(data_path):
        if '500Hz' not in test:
            continue
        test_path = os.path.join(data_path, test)
        amps = []
        list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(test_path, x)),
                    os.listdir(test_path) ) )
        # for mic in os.listdir(test_path):
        for mic in list_of_files:
            if mic[-4:] != '.wav':
                continue
            mic_file = os.path.join(test_path, mic)
            y, sr = sf.read(mic_file)
            amps.append(y)
        output_path = os.path.join(test_path, 'magnitudes')
        save_plot(amps, output_path)