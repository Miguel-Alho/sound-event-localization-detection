#!/usr/bin/env python3

import rospy
from sound_generator import SoundGenerator

if __name__ == '__main__':
    rospy.init_node('sound_generator', anonymous=True)
    sound_generator = SoundGenerator()
    sound_generator.run()