#!/usr/bin/env python3

import rospy
from sound_loader import SoundLoader

if __name__ == '__main__':
    rospy.init_node('sound_loader', anonymous=True)
    sound_loader = SoundLoader()