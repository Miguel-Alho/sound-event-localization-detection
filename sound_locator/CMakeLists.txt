cmake_minimum_required(VERSION 3.0.2)
project(sound_locator)

find_package(catkin REQUIRED rospy sound_msgs sound_utils)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

catkin_package(
    DEPENDS sound_msgs sound_utils
)