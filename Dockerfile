#DOES NOT WORK YET

#Pulls base Docker contatiner
FROM osrf/ros:humble-desktop-full


#Downloads additional packages
RUN sudo apt-get upgrade
RUN sudo apt-get update
RUN sudo apt -y install python3-pip
RUN pip3 install setuptools==58.2.0
RUN sudo apt -y install vim
RUN export TZ='America/Denver'


#Adds contents of folder into container
COPY ./src /root/src
COPY ./bags /root/bags
COPY ./build /root/build
COPY ./install /root/install
COPY ./build.sh /root/build.sh
RUN chmod +x /root/build.sh

#Creates Docker contatiner
#RUN docker run -it   --cap-add=SYS_PTRACE  --net host --ipc host  --privileged   -e DISPLAY   -e XAUTHORITY   -v /tmp/.X11-unix:/tmp/.X11-unix  --name humble_devel osrf/ros:humble-desktop-full


