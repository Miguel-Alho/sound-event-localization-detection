import rospy

from sound_msgs.msg import *


class SoundReceiverBase:
    def __init__(self):
        self.raw_data_pub = rospy.Publisher('raw_data', SoundSample, queue_size=10)
    
    def run(self):
        try:
            index = 0
            while not rospy.is_shutdown():
                msg:SoundSample = self.get_data_to_publish(index)
                self.raw_data_pub.publish(msg)
                index += 1

        except rospy.ROSInterruptException:
            pass
    
    def get_data_to_publish(self):
        pass