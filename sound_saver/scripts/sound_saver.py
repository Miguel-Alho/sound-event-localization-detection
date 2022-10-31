#!/usr/bin/env python3

import os
import math
import rospy
import sounddevice as sd
import soundfile as sf
import multiprocessing

import sound_utils
from utils import *
from sound_msgs.msg import *
# from sound_source import SoundSource
from microphone import Microphone
# from microphone_base import MicrophoneBase


DATA_PATH = '/home/miguel/sound_classification/src/MobileRobots/sound_tools/sound_saver/sounds'
RECORD_TIME = 2.0

class SoundSaver():
    def __init__(self):
        self.sound_sources = parse_sources_base()
        self.mics = Microphone.ParseMicrophones()
        self.mic_data = []
        n_mics = len(self.mics)
        n_srcs = len(self.sound_sources)
        mics_dist = round( math.dist((self.mics[0].x, self.mics[0].y, self.mics[0].z), (self.mics[1].x, self.mics[1].y, self.mics[1].z)), 2)
        index = 0
        last_folder = ''
        coords = '2D'
        for src in self.sound_sources:
            file = src.file[:-4]
            if IN_3D:
                coord = '({},{},{})'.format(src.x, src.y, src.z)
                coords = '3D'
            else:
                coord = '({},{})'.format(src.x, src.y)
            last_folder += '{}_{}_{}_'.format(src.type, file, coord)
        self.folder_path = os.path.join(DATA_PATH, ROOM, '{}_{}mics_{}cm'.format(coords, n_mics, int(mics_dist*100)), '{}_src'.format(n_srcs), last_folder+str(index))
        while(True):
            if os.path.exists(self.folder_path):
                index += 1
                self.folder_path = self.folder_path[:-1] + str(index)
            else:
                break
        os.makedirs(self.folder_path)

    
    def run(self):
        for mic in self.mics:
            mic.read()
            data = []
            self.mic_data.append(data)

        record_iter = RECORD_TIME/TIME_FRAME
        if record_iter % 1 != 0:
            rospy.logwarn('The RECORD_TIME is not a multiple of TIME_FRAME. The output will be slightly shorter.')
        rospy.loginfo('\n\nStarted recording {} seconds.'.format(RECORD_TIME))
        for _ in range(int(record_iter)):
            self.get_data_to_record()

        # close_mics(self.mics)
        for mic in self.mics:
            mic.stream.stop()
            rospy.loginfo(mic.id + ': stream stopped')
            mic.stream.close()
            rospy.loginfo(mic.id + ': stream closed')

        for i, data in enumerate(self.mic_data):
            print(len(data))
            file_name = '{}.wav'.format(self.mics[i].id)
            file_path = os.path.join( self.folder_path, file_name )
            sf.write(file_path, data, SAMPLE_RATE)

    
    def get_data_to_record(self):
        for mic in self.mics:
            mic.start_recording = True
        for i, mic in enumerate(self.mics):
            self.mic_data[i].extend(mic.get_real_data_to_record())