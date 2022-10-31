cmake_minimum_required(VERSION 3.0.2)
project(sound_spec)

find_package(catkin REQUIRED rospy sound_msgs sound_utils)

catkin_python_setup()

catkin_package(
    DEPENDS sound_msgs sound_utils
)