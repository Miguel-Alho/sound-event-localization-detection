import rospy
import soundfile as sf
import numpy as np
import rospkg
import scipy.signal as sci

from utils import SAMPLE_RATE, FRAME_LENGTH
from sound_msgs.msg import *
from sound_source_base import SoundSourceBase


class SoundSource(SoundSourceBase):
    def __init__(self, params):
        super().__init__(params)
        self.data = []
    
    def process(self):
        rospy.loginfo('Loading Generated Audio from ' + self.file)
        try:
            self.read_raw_data_from_file()
            rospy.loginfo('Successfully loaded sound file ' + self.file)
        except:
            rospy.logerr('Failed to load sound file ' + self.file)
            pass

    def read_raw_data_from_file(self):
        rp = rospkg.RosPack()
        package_path = rp.get_path('sound_generator') + '/' + 'sounds'
        data, sr = sf.read(package_path + '/' + self.file)
        signal_length = len(data)
        # If the audio file has 2 channels consider just the first one
        if len(data.shape) != 1:
            data = data[:, 0]
        # Resample to the desired sample rate
        if sr != SAMPLE_RATE:
            rospy.logerr('Original audio lenght: {}'.format(signal_length))
            sample_size = int((SAMPLE_RATE/sr) * signal_length)
            data = sci.resample(data, sample_size)
            rospy.logerr('Audio length after resample: {}'.format(signal_length))
        # Divide the audio file in frames
        n_frames = signal_length // FRAME_LENGTH
        if signal_length > FRAME_LENGTH*n_frames:
            difference = FRAME_LENGTH*(n_frames+1) - signal_length
            zero_pad = np.zeros(difference)
            signal = np.append(data, zero_pad)
        # rospy.logerr(n_frames)
        # rospy.logerr(FRAME_LENGTH)
        # rospy.logerr(signal_length)
        rospy.logerr(len(signal))
        for i in range(n_frames+1):
            self.data.append(signal[i*FRAME_LENGTH : (i+1)*FRAME_LENGTH])

        
    @staticmethod
    def ParseSoundSources():
        # params = rospy.get_param('~sound_sources', []) # the tilt is for private sound sources
        params = rospy.get_param('sound_sources', [])
        if len(params) == 0:
            rospy.logerr('Failed to detect any audio source')
            return None
        sound_sources = []
        for param in params:
            sound_source = SoundSource(param)
            sound_sources.append(sound_source)
        return sound_sources