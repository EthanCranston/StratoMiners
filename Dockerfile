#DOES NOT WORK YET

#Pulls base Docker contatiner
FROM osrf/ros:humble-desktop-full

#Adds contents of folder into container
COPY ./src ./

#Creates Docker contatiner
#RUN docker run -it   --cap-add=SYS_PTRACE  --net host --ipc host  --privileged   -e DISPLAY   -e XAUTHORITY   -v /tmp/.X11-unix:/tmp/.X11-unix  --name humble_devel osrf/ros:humble-desktop-full

#COPY from outside to inside container