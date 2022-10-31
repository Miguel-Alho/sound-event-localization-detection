#!/usr/bin/env python3

import rospy

import sound_utils
from sound_msgs.msg import *
from microphone import Microphone
from receiver_base import SoundReceiverBase


class SoundReceiver(SoundReceiverBase):
    def __init__(self):
        super().__init__()
        self.mics = Microphone.ParseMicrophones()
    
    def run(self):
        for mic in self.mics:
            mic.read()
        # rospy.spin()        
        super().run()
        for mic in self.mics:
            mic.stream.stop()
            rospy.loginfo(mic.id + ': stream stopped')
            mic.stream.close()
            rospy.loginfo(mic.id + ': stream closed')
    
    def get_data_to_publish(self, index):
        for mic in self.mics:
            mic.start_recording = True
        mic_data = SoundSample()
        for mic in self.mics:
            mic_data.frame_arrays.append(mic.get_real_data_to_publish())
        return mic_data