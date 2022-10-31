from typing import List
import numpy as np
import math
from scipy.spatial.transform import Rotation
import rospy
import matplotlib.pyplot as plt
# from microphone import Microphone
from microphone_base import MicrophoneBase
from sound_source_base import SoundSourceBase


ROOM = rospy.get_param('room')
IN_3D = rospy.get_param('in_3D')
DATA_NUM = rospy.get_param('data_num')
ROOM_SIZE = rospy.get_param('room_size')
# MIN_ENERGY_THRESHOLD = rospy.get_param('min_energy_threshold')
MIN_AMP_RATIO_THRESHOLD = rospy.get_param('min_amp_ratio_threshold')
MIN_AMP_THRESHOLD = rospy.get_param('min_amp_threshold')
# MIN_AMP_DIF_THRESHOLD = rospy.get_param('min_amp_dif_threshold')
MIC_AREA_THRESHOLD = rospy.get_param('mic_area_threshold')
DIST_THRESHOLD = rospy.get_param('dist_threshold')
FREQ_RANGE = rospy.get_param('freq_range')
MAX_ZERO_DIF_THRESHOLD = rospy.get_param('max_zero_dif_threshold')
TIME_LIMIT = rospy.get_param('time_limit')
PRA = rospy.get_param('pra')
POLAR_PATTERN_COEF = rospy.get_param('polar_pattern_coef')
PLOT_CIRCLES = rospy.get_param('plot_circles')
PLOT_MAP = rospy.get_param('plot_map')
PLOT_CURRENT_MAP = rospy.get_param('plot_current_map')
ADD_TO_MICS_AREA = rospy.get_param('add_to_mics_area')
MICS_CENTER = rospy.get_param('mics_center')
MAP_OPTION = rospy.get_param('map_option')

FRAME_SIZE = rospy.get_param('frame_size')
HOP_LENGTH = rospy.get_param('hop_length')
N_CHANNELS = len(rospy.get_param('mics'))
TIME_FRAME = rospy.get_param('time_frame')
SAMPLE_RATE = rospy.get_param('sample_rate')
FRAME_LENGTH = int(TIME_FRAME * SAMPLE_RATE)
CLASSIFIER = rospy.get_param('classifier')
HEARING_THRESHOLD = 1*10**-12
# HEARING_THRESHOLD = np.max



def parse_microphones_base():
    params = rospy.get_param('mics')
    mic_array = []
    for param in params:
        mic = MicrophoneBase(param)
        mic_array.append(mic)
    return mic_array


def parse_sources_base():
    params = rospy.get_param('sound_sources')
    src_array = []
    for param in params:
        src = SoundSourceBase(param)
        src_array.append(src)
    return src_array


# def close_mics(mics: List[Microphone]):
#     for mic in mics:
#         mic.stream.stop()
#         rospy.loginfo(mic.id + ': stream stopped')
#         mic.stream.close()
#         rospy.loginfo(mic.id + ': stream closed')


def get_ith_min(matrix):
    n_pixels = matrix.shape[0] * matrix.shape[1]
    min_index = int(n_pixels*MAX_ZERO_DIF_THRESHOLD)
    array = matrix.reshape(n_pixels)
    min = np.partition(array, min_index)[min_index]
    return min


def get_mic_direction(mic_quaternions):
    r = Rotation.from_quat(mic_quaternions)
    return r.as_matrix()[:,0]

def angle_2D_vectors(A, B, C, D):
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    x4, y4 = D
    Xa = x2 - x1
    Ya = y2 - y1
    Xb = x4 - x3
    Yb = y4 - y3
    return (Xa*Xb + Ya*Yb) / (np.sqrt(Xa**2 + Ya**2) * np.sqrt(Xb**2 + Yb**2)) # cos_theta

def angle_3D_vectors(A, B, C, D):
    x1, y1, z1 = A
    x2, y2, z2 = B
    x3, y3, z3 = C
    x4, y4, z4 = D
    Xa = x2 - x1
    Ya = y2 - y1
    Za = z2 - z1
    Xb = x4 - x3
    Yb = y4 - y3
    Zb = z4 - z3
    return (Xa*Xb + Ya*Yb + Za*Zb) / (np.sqrt(Xa**2 + Ya**2 + Za**2) * np.sqrt(Xb**2 + Yb**2 + Zb**2)) # cos_theta

def polar_patter_func(coord_mic, coord_src, direct):
    if IN_3D: cos_theta = angle_3D_vectors(coord_mic, coord_src, (0,0,0), direct)
    else: cos_theta = angle_2D_vectors(coord_mic[:2], coord_src[:2], (0,0), direct[:2])
    return POLAR_PATTERN_COEF + (1-POLAR_PATTERN_COEF) * cos_theta

def create_map_2D():
    a = np.linspace( 0 , ROOM_SIZE[0] , DATA_NUM*ROOM_SIZE[0] )
    b = np.linspace( 0 , ROOM_SIZE[1] , DATA_NUM*ROOM_SIZE[1] )
    return np.meshgrid( a, b )

def create_map_3D():
    a = np.linspace( 0 , ROOM_SIZE[0] , DATA_NUM*ROOM_SIZE[0] )
    b = np.linspace( 0 , ROOM_SIZE[1] , DATA_NUM*ROOM_SIZE[1] )
    c = np.linspace( 0 , ROOM_SIZE[2] , DATA_NUM*ROOM_SIZE[2] )
    return np.meshgrid( a, b, c )


