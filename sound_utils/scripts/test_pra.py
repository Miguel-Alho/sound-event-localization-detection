#!/usr/bin/env python3

import os
import soundfile as sf
import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)

SAMPLE_RATE = 44100


room = pra.ShoeBox([8,8], fs=SAMPLE_RATE, max_order=0)

dir_1 = CardioidFamily(
    orientation=DirectionVector(azimuth=-135, degrees=True),
    pattern_enum=DirectivityPattern.CARDIOID,
)
dir_2 = CardioidFamily(
    orientation=DirectionVector(azimuth=135, degrees=True),
    pattern_enum=DirectivityPattern.CARDIOID,
)
dir_3 = CardioidFamily(
    orientation=DirectionVector(azimuth=45, degrees=True),
    pattern_enum=DirectivityPattern.CARDIOID,
)
dir_4 = CardioidFamily(
    orientation=DirectionVector(azimuth=-45, degrees=True),
    pattern_enum=DirectivityPattern.CARDIOID,
)

mic_locs = np.c_[[3.6, 3.6],
                [3.6, 4.4],
                [4.4, 4.4],
                [4.4, 3.6]]

room.add_microphone_array(mic_locs, directivity=[dir_1, dir_2, dir_3, dir_4])

src_loc = [4, 1]
src_signal, sr = sf.read('/home/miguel/sound_classification/src/MobileRobots/sound_tools/sound_generator/sounds/500Hz.wav')
room.add_source(src_loc, signal=src_signal)

room.simulate()

# room.mic_array.to_wav( '/home/miguel/Documents/test.wav', norm=True, bitdepth=np.int16 )

signals = room.mic_array.signals

folder_path = '/home/miguel/sound_classification/src/MobileRobots/sound_tools/sound_saver/sounds'
for i, signal in enumerate(signals):
    file_name = 'src{}.wav'.format(i)
    file_path = os.path.join( folder_path, file_name )
    sf.write(file_path, signal, SAMPLE_RATE)