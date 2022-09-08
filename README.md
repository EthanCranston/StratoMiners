# StratoMiners
Field session (CSCI-370) project for Stratom

##Getting Started
- Use Ubuntu 6.1
- Install docker as documented [here](https://docs.docker.com/desktop/install/linux-install/)
- run `sudo usermod -aG docker {username}` and restart Ubuntu. This sets permissions so `sudo` doesn't have to be used in docker commands
- run `git clone https://github.com/EthanCranston/StratoMiners.git` to pull the repo.
- run `docker build . -t strato-miners-container` in the StratoMiners directory to build the image from the Dockerfile
- run `docker run -it   --cap-add=SYS_PTRACE  --net host --ipc host  --privileged   -e DISPLAY   -e XAUTHORITY   -v /tmp/.X11-unix:/tmp/.X11-unix  --name StratoMinersDev strato-miners-container`

## Build
Run the "build.sh" script in the main directory of the workspace


