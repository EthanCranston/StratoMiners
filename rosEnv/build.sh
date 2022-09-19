#!/bin/bash

#first we build all the packages
colcon build --packages-skip darknet_ros

source ./install/setup.bash
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
source /opt/ros/humble/setup.bash
source /colcon_workspace/install/local_setup.bash