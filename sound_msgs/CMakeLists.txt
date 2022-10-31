cmake_minimum_required(VERSION 3.0.2)
project(sound_msgs)

set(MSG_DEPS actionlib_msgs std_msgs)

find_package(catkin REQUIRED COMPONENTS message_generation ${MSG_DEPS})

set(MSG_FILES
  SoundPoint.msg
  SoundFrame.msg
  SoundFrameArray.msg
  SoundSample.msg
  SoundSampleArray.msg
)

set(SRV_FILES)

set(ACT_FILES
  Classifier.action
)

add_action_files(DIRECTORY action FILES ${ACT_FILES})
add_message_files(DIRECTORY msg FILES ${MSG_FILES})
#add_service_files(DIRECTORY srv FILES ${SRV_FILES})

generate_messages(DEPENDENCIES ${MSG_DEPS})

catkin_package(CATKIN_DEPENDS message_runtime ${MSG_DEPS})