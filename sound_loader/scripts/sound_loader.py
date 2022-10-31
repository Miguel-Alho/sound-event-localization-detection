#!/usr/bin/env python3

from tkinter import filedialog
from tkinter import *
import os
import rospy
import rospkg
import soundfile as sf

import sound_utils
from utils import *
from sound_msgs.msg import *
from microphone import Microphone
from receiver_base import SoundReceiverBase


class SoundLoader(SoundReceiverBase):
    def __init__(self):
        super().__init__()
        self.mics = Microphone.ParseMicrophones()
        self.frequency = rospy.get_param('frequency', 1)
        self.rate = rospy.Rate(self.frequency)
        rp = rospkg.RosPack()
        data_path = rp.get_path('sound_saver') + '/' + 'sounds' + '/living_room/2D_4mics_80cm/1_src'
        self.ask_path(data_path)
        rospy.loginfo(self.path)
        self.load_files()
        super().run()


    def load_files(self):
        mic_index = 0
        list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(self.path, x)),
                    os.listdir(self.path) ) )
        for mic_file in list_of_files:
            if mic_file[-4:] != '.wav':
                continue
            print(mic_file)
            mic_path = os.path.join(self.path, mic_file)
            data, sr = sf.read(mic_path)
            n_frames = len(data) // FRAME_LENGTH
            for i in range(n_frames):
                data_frame = data[i*FRAME_LENGTH : (i+1)*FRAME_LENGTH]
                self.mics[mic_index].raw_data.put(data_frame)
            mic_index += 1


    def ask_path(self, data_path):
        root = Tk()
        root.withdraw()
        self.path = filedialog.askdirectory(title='Choose a folder with mics audio files', initialdir=data_path)


    def get_data_to_publish(self, index):
        self.rate.sleep()
        mic_data = SoundSample()
        for mic in self.mics:
            mic_data.frame_arrays.append(mic.get_real_data_to_publish())
        return mic_data