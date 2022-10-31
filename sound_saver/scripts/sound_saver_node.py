#!/usr/bin/env python3

import rospy
from sound_saver import SoundSaver

if __name__ == '__main__':
    rospy.init_node('sound_saver', anonymous=True)
    sound_saver = SoundSaver()
    sound_saver.run()