import rospy

import sound_utils
from sound_msgs.msg import *
from microphone import Microphone
from sound_source import SoundSource
from receiver_base import SoundReceiverBase


class SoundGenerator(SoundReceiverBase):
    def __init__(self):
        super().__init__()
        self.sound_sources = SoundSource.ParseSoundSources()
        self.mics = Microphone.ParseMicrophones()
    
    def run(self):
        if not self.read_audio_source():
            rospy.logfatal('Failed to read audio file')
            return
        if not self.generate_mic_audio_source():
            rospy.logfatal('Failed to generate mics audios')
            return
        super().run()
    
    def read_audio_source(self):
        for source_audio in self.sound_sources:
            source_audio.process()
        return True
    
    def generate_mic_audio_source(self):
        for mic in self.mics:
            mic.simulate_mic(self.sound_sources)
        return True
    
    def get_data_to_publish(self, index):
        mic_data = SoundSample()
        for mic in self.mics:
            mic_data.frame_arrays.append(mic.get_data_to_publish(index))         
        return mic_data