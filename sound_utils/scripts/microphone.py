from typing import List
import rospy
import math
import sounddevice as sd
# import pyaudio
import scipy.signal as sci
import queue
import threading
import time
import pyroomacoustics as pra

import sound_utils
from utils import *
from sound_msgs.msg import *
from sound_source import SoundSource
from microphone_base import MicrophoneBase


class Microphone(MicrophoneBase):
    def __init__(self, params):
        super().__init__(params)
        self.data = []  # SounfFrames array
        self.raw_data = queue.Queue()
        self.start_recording = False
        self.count = 0
        self.input_device_index = 0

    def read_thread(self):
        while True:
            time.sleep(0.01) # FIXME: If this is not here, it won't work
            if self.start_recording:
                # rospy.logerr(FRAME_LENGTH)
                # rospy.logerr(self.id + ': Start recording')
                data, ov = self.stream.read(FRAME_LENGTH)
                # rospy.loginfo(self.id + ': Finish recording')
                data = data.reshape(FRAME_LENGTH)
                self.raw_data.put(data)
                if ov:
                    rospy.logerr('Input overflow')
                self.start_recording = False

    def read_thread2(self):
        frame_length = SAMPLE_RATE*5
        while True:
            time.sleep(0.01) # FIXME: If this is not here, it won't work
            if self.start_recording:
                # rospy.logerr(FRAME_LENGTH)
                # rospy.logerr(self.id + ': Start recording')
                data, ov = self.stream.read(frame_length)
                self.stream.stop()
                self.stream.close()
                # rospy.loginfo(self.id + ': Finish recording')
                data = data.reshape(frame_length)
                self.raw_data.put(data)
                if ov:
                    rospy.logerr('Input overflow')
                self.start_recording = False


    def read(self):
        device_id = self.id[-1]
        device_name = 'hw:' + device_id
        self.stream = sd.InputStream(device=device_name, channels=1, blocksize=FRAME_LENGTH)
        self.stream.start()
        threading.Thread(target=self.read_thread, daemon=True).start()


    def get_data_to_publish(self, index):
        aux_index = index % len(self.data)
        # rospy.logerr(self.data[aux_index])
        return self.data[aux_index]


    def get_real_data_to_publish(self):
        # rospy.loginfo(self.id)
        # rospy.loginfo(self.raw_data.qsize())
        data = self.raw_data.get()
        return self.stft_of_raw_data(data)


    def get_real_data_to_record(self):
        return self.raw_data.get()


    @staticmethod
    def ParseMicrophones():
        params = rospy.get_param('mics')
        mic_array = []
        for param in params:
            mic = Microphone(param)
            mic_array.append(mic)
        return mic_array


    def get_n_samples(self, sources: List[SoundSource]):
        src_data_length = 9999999999999
        for src in sources:
            if len(src.data) < src_data_length:
                src_data_length = len(src.data)
        return src_data_length


    def simulate_mic(self, sources: List[SoundSource]):
        '''
        Each source has a data list of amplitudes which need to be summed and converted to stft
        '''
        n_samples = self.get_n_samples(sources)
        coord_mic = np.array([self.x, self.y, self.z])
        for i in range(n_samples):
            sound_frame_array = SoundFrameArray()
            framesDict = dict()

            for src in sources:
                # If this source audio ended, continue to next source
                if i >= len(src.data):
                    continue
                coord_src = np.array([src.x, src.y, src.z])
                dist = math.dist(coord_mic, coord_src)
                direct = get_mic_direction([self.rx, self.ry, self.rz, self.rw])
                if PRA:
                    coord = (coord_src - coord_mic) / dist # if mic is centered in (0,0) the src is in 'coord' on the unitary circle
                    polar_pattern = pra.cardioid_func(x=coord, direction=direct, coef=POLAR_PATTERN_COEF, magnitude=True)
                else: polar_pattern = polar_patter_func(coord_mic, coord_src, direct)

                # Obtain Frequency, Time and Amplitudes in Arrays
                tmp_freq_array, tmp_time_array, tmp_complex_amplitude_array = sci.stft(src.data[i], SAMPLE_RATE, nperseg=FRAME_SIZE, noverlap=HOP_LENGTH)
                tmp_amplitude_array = np.abs(tmp_complex_amplitude_array)

                for bin_index, _ in enumerate(tmp_time_array):
                    if not bin_index in framesDict.keys():
                        framesDict[bin_index] = dict()
                    for f_index, f in enumerate(tmp_freq_array):
                        amp = polar_pattern * tmp_amplitude_array[f_index, bin_index] / dist
                        if not f in framesDict[bin_index].keys():
                            framesDict[bin_index][f] = 0
                        framesDict[bin_index][f] += amp

            # Convert sample to msg and append to self.data
            for bin_index in framesDict.keys():
                sound_frame = SoundFrame()
                sound_frame.mic_id = self.id
                for freq in framesDict[bin_index].keys():
                    point = SoundPoint()
                    point.frequency = freq
                    point.amplitude = framesDict[bin_index][freq]
                    sound_frame.points.append(point)
                sound_frame_array.frames.append(sound_frame)
            self.data.append(sound_frame_array)


    def stft_of_raw_data(self, data):
        stft = SoundFrameArray()
        tmp_freq_array, tmp_time_array, tmp_complex_amplitude_array = sci.stft(data, SAMPLE_RATE, nperseg=FRAME_SIZE, noverlap=HOP_LENGTH)
        tmp_amplitude_array = np.abs(tmp_complex_amplitude_array) # magnitude (1 + FRAME_SIZE/2, 2 + len(data)/HOP_LENGTH)
        for bin_index, bin_time in enumerate(tmp_time_array):
            sound_frame = SoundFrame()
            sound_frame.timestamp = rospy.Time.from_sec(bin_time)
            sound_frame.mic_id = self.id

            for f_index, f in enumerate(tmp_freq_array):
                point = SoundPoint()
                point.amplitude = tmp_amplitude_array[f_index, bin_index]
                point.frequency = f
                sound_frame.points.append(point)

            stft.frames.append(sound_frame)
        return stft
