#!/usr/bin/env python3

import rospy
from sound_receiver import SoundReceiver

if __name__ == '__main__':
    rospy.init_node('sound_receiver', anonymous=True)
    sound_receiver = SoundReceiver()
    sound_receiver.run()