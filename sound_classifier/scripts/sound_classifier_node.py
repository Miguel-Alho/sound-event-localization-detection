#!/usr/bin/env python3

import rospy
from sound_classifier import SoundClassifier

if __name__ == '__main__':
    rospy.init_node('sound_classifier', anonymous=True)
    sound_generator = SoundClassifier()
    sound_generator.run()