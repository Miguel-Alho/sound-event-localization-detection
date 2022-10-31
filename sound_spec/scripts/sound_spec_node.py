#!/usr/bin/env python3

import rospy
from sound_spec import SoundSpec

if __name__ == '__main__':
    rospy.init_node('sound_spec', anonymous=True)
    sound_spec = SoundSpec()
    sound_spec.run()