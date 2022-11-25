# Pulls OpenCV-enabled docker container
FROM dkimg/opencv:4.5.5-ubuntu

# Pulls base Docker container
# syntax=docker/dockerfile:1.0.0-experimental
###########################################
# Base image 
###########################################
FROM nvidia/cuda:11.7.0-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install language
RUN apt-get update && apt-get install -y \
  locales \
  && locale-gen en_US.UTF-8 \
  && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/*
ENV LANG en_US.UTF-8

# Install timezone
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
  && export DEBIAN_FRONTEND=noninteractive \
  && apt-get update \
  && apt-get install -y tzdata \
  && dpkg-reconfigure --frontend noninteractive tzdata \
  && rm -rf /var/lib/apt/lists/*

# Install ROS2
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    sudo \
  && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
  && apt-get update && apt-get install -y \
    ros-humble-ros-base \
    python3-argcomplete \
  && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=humble \
  AMENT_PREFIX_PATH=/opt/ros/humble \
  COLCON_PREFIX_PATH=/opt/ros/humble \
  LD_LIBRARY_PATH=/opt/ros/humble/lib \
  PATH=/opt/ros/humble/bin:$PATH \
  PYTHONPATH=/opt/ros/humble/lib/python3.8/site-packages \
  ROS_PYTHON_VERSION=3 \
  ROS_VERSION=2 
ENV DEBIAN_FRONTEND=

###########################################
#  Develop image 
###########################################
FROM base AS dev

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
  bash-completion \
  build-essential \
  cmake \
  gdb \
  git \
  python3-argcomplete \
  python3-colcon-common-extensions \
  python3-pip \
  python3-rosdep \
  python3-vcstool \
  vim \
  wget \
  python3-autopep8 \
  && rm -rf /var/lib/apt/lists/* \
  && rosdep init || echo "rosdep already initialized"

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create a non-root user
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  # [Optional] Add sudo support for the non-root user
  && apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  # Cleanup
  && rm -rf /var/lib/apt/lists/* \
  && echo "source /usr/share/bash-completion/completions/git" >> /home/$USERNAME/.bashrc \
  && echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then source /opt/ros/${ROS_DISTRO}/setup.bash; fi" >> /home/$USERNAME/.bashrc
ENV DEBIAN_FRONTEND=

###########################################
#  Full image 
###########################################
FROM dev AS full

ENV DEBIAN_FRONTEND=noninteractive
# Install the full release
RUN apt-get update && apt-get install -y \
  ros-humble-desktop \
  ros-humble-ament-lint \
  ros-humble-launch-testing \
  ros-humble-launch-testing-ament-cmake \
  ros-humble-launch-testing-ros \
  ros-humble-rmw-cyclonedds-cpp \
  ros-humble-ament-clang-format \
  && rm -rf /var/lib/apt/lists/* \
  && pip install colcon-mixin \
  && colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml \
  && colcon mixin update default
ENV DEBIAN_FRONTEND=

###########################################
#  Full+Gazebo image 
###########################################
FROM full AS gazebo

ENV DEBIAN_FRONTEND=noninteractive
# Install gazebo
RUN apt-get update && apt-get install -y \
  ros-humble-gazebo* \
  && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=

###########################################
#  Full+Gazebo+Nvidia image 
###########################################

FROM gazebo AS gazebo-nvidia

################
# Expose the nvidia driver to allow opengl 
# Dependencies for glvnd and X11.
################
RUN apt-get update \
 && apt-get install -y -qq --no-install-recommends \
  libglvnd0 \
  libgl1 \
  libglx0 \
  libegl1 \
  libxext6 \
  libx11-6 \
  python3-vcstool \
  git \
  software-properties-common \
  libceres-dev \
  nlohmann-json3-dev

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV QT_X11_NO_MITSHM 1
ENV CUDACXX=/usr/local/cuda-11.7/bin/nvcc \
  CMAKE_CUDA_COMPILER=/usr/local/cuda-11.7/bin/nvcc \
  CUDA_NVCC_EXECUTABLE=/usr/local/cuda-11.7/bin/nvcc \
  GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH/opt/ros/humble/lib:

###########################################
#  Post-stage installation
#  Install config-specific tools 
###########################################

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

###########################################
#  Build complete
#  Set ROS vars
###########################################

# Set ROS2 Humble vars
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV ROS_DISTRO=humble
ENV AMENT_PREFIX_PATH=/opt/ros/humble
ENV COLCON_PREFIX_PATH=/opt/ros/humble
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib:/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib 
ENV PATH=/opt/ros/humble/bin:$PATH:/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin 
ENV PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages 
ENV ROS_PYTHON_VERSION=3
ENV ROS_VERSION=2
ENV DEBIAN_FRONTEND=

# Creates Docker contatiner
# RUN docker run -it   --cap-add=SYS_PTRACE  --net host --ipc host  --privileged   -e DISPLAY   -e XAUTHORITY   -v /tmp/.X11-unix:/tmp/.X11-unix  --name humble_devel osrf/ros:humble-desktop-full