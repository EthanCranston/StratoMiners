# Pulls OpenCV-enabled docker container
FROM dkimg/opencv:4.5.5-ubuntu

# Pulls base Docker contatiner
FROM osrf/ros:humble-desktop-full

# Installs apt-utils
RUN sudo apt-get upgrade -y
RUN sudo apt-get update -y
RUN sudo apt-get install dialog apt-utils -y

# Downloads additional packages
RUN sudo apt -y install python3-pip
RUN pip3 install setuptools==58.2.0
RUN pip3 install scikit-learn
RUN sudo apt -y install vim
RUN sudo apt -y install ros-humble-rmw-cyclonedds-cpp
RUN sudo apt -y install libpcl-dev
RUN sudo apt-get install -y libboost-all-dev
RUN sudo apt-get install libc6-dbg gdb valgrind -y

# Exports necessary variables
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp 
ENV TZ='America/Denver'

# Sets up work environment and copies rosEnv contents
COPY . /rosEnv
WORKDIR /rosEnv
ENV HOME /rosEnv

# Creates Docker contatiner
# RUN docker run -it   --cap-add=SYS_PTRACE  --net host --ipc host  --privileged   -e DISPLAY   -e XAUTHORITY   -v /tmp/.X11-unix:/tmp/.X11-unix  --name humble_devel osrf/ros:humble-desktop-full