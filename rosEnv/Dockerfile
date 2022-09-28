#Pulls base Docker contatiner
FROM osrf/ros:humble-desktop-full

#Installs apt-utils
RUN sudo apt-get upgrade -y
RUN sudo apt-get update -y
RUN sudo apt-get install dialog apt-utils -y

#Downloads additional packages
RUN sudo apt -y install python3-pip
RUN pip3 install setuptools==58.2.0
RUN pip3 install opencv-python
RUN sudo apt -y install vim
RUN sudo apt -y install emacs
RUN export TZ='America/Denver'

#Installs OpenCV so that each container doesn't have to that individually
RUN mkdir opencv
RUN cd opencv
RUN sudo apt install -y cmake g++ wget unzip
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
RUN unzip opencv.zip
RUN mkdir -p build && cd build
RUN cmake  ../opencv-4.x
RUN cmake --build .

# Installs boost and necessary requirements
RUN cd ../..
RUN sudo apt-get install -y libboost-all-dev

# Sets up work environment and copies rosEnv contents
COPY . .
WORKDIR /rosEnv
RUN chmod +x /build.sh
RUN chmod +x /install.sh

#Creates Docker contatiner
#RUN docker run -it   --cap-add=SYS_PTRACE  --net host --ipc host  --privileged   -e DISPLAY   -e XAUTHORITY   -v /tmp/.X11-unix:/tmp/.X11-unix  --name humble_devel osrf/ros:humble-desktop-full
