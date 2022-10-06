#!/bin/bash

# Source all packages
source ./install/setup.bash
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
source /opt/ros/humble/setup.bash
source /rosEnv/install/local_setup.bash

# Create alias for the launch file
alias start_node="ros2 launch /rosEnv/src/launch/launch.py"