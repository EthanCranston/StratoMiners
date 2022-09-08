#!/usr/bin/env bash

# Check if the user has the docker group, add it and stop w/ message if not
#if !(id -nG "$USER" | grep -q '\bdocker\b'); then
#    echo "User $USER does not have the docker group. Adding, please restart linux and try again."
#    sudo usermod -aG docker $USER
#fi

container_name="dev_container"

# Add xhosts
xhost +local:root

# Start the humble_devel docker 
alias start_dev="docker run -it --cap-add=SYS_PTRACE --net host --ipc host --privileged -e DISPLAY -e XAUTHORITY -v /tmp/.X11-unix:/tmp/.X11-unix --name $container_name strato-miners-container"

# General utilites for restart, remove 
alias open_dev="docker start -i $container_name"
alias rm_dev="docker rm $container_name"
alias build_dev="docker build . -t strato-miners-container"

