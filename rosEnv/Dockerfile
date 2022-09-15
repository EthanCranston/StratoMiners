#Pulls base Docker contatiner
FROM osrf/ros:humble-desktop-full

#Installs apt-utils
RUN sudo apt-get update -y
RUN sudo apt-get install dialog apt-utils -y

#Downloads additional packages
RUN sudo apt-get upgrade -y
RUN sudo apt-get update -y
RUN sudo apt -y install python3-pip
RUN pip3 install setuptools==58.2.0
RUN pip3 install opencv-python
RUN sudo apt -y install vim
RUN sudo apt -y install emacs
RUN export TZ='America/Denver'

#Installs boost and necessary requirements
RUN cd ../..
RUN sudo apt-get install -y libboost-all-dev

#Adds contents of folder into container
COPY ./src /root/src
COPY ./bags /root/bags
COPY ./scripts /root/scripts
COPY ./build.sh /root/build.sh
RUN chmod +x /root/build.sh
RUN chmod +x /root/scripts/darknet.sh
RUN chmod +x /root/scripts/opencv.sh

# Sets up work environment
COPY . .
WORKDIR /rosEnv
RUN chmod +x /build.sh
RUN chmod +x /scripts/darknet.sh
RUN chmod +x /scripts/opencv.sh

#Creates Docker contatiner
#RUN docker run -it   --cap-add=SYS_PTRACE  --net host --ipc host  --privileged   -e DISPLAY   -e XAUTHORITY   -v /tmp/.X11-unix:/tmp/.X11-unix  --name humble_devel osrf/ros:humble-desktop-full
