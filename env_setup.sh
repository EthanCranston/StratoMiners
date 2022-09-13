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
alias start_dev="docker compose up -d && docker attach $container_name"

# General utilites for restart, remove 
alias open_dev="docker start -i $container_name"
alias rm_dev="docker rm $container_name"
alias build_dev="docker build . -t strato-miners-container"

