#!/usr/bin/env python3

import rospy
from sound_locator import SoundLocator

if __name__ == '__main__':
    rospy.init_node('sound_locator', anonymous=True)
    sound_locator = SoundLocator()
    sound_locator.run()