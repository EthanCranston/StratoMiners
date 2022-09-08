# StratoMiners
Field session (CSCI-370) project for Stratom

## Getting Started
- Use Ubuntu 22.04 LTS
- Install docker as documented [here](https://docs.docker.com/desktop/install/linux-install/)
- run `sudo usermod -aG docker {username}` and restart Ubuntu. This sets permissions so `sudo` doesn't have to be used in docker commands
- run `git clone https://github.com/EthanCranston/StratoMiners.git` to pull the repo.
Then, either source env_setup.sh, or
- run `docker build . -t strato-miners-container` in the StratoMiners directory to build the image from the Dockerfile
- run `xhost +local:root` in the StratoMiners directory to enable local GUI access
- run `docker run -it   --cap-add=SYS_PTRACE  --net host --ipc host  --privileged   -e DISPLAY   -e XAUTHORITY   -v /tmp/.X11-unix:/tmp/.X11-unix  --name StratoMinersDev strato-miners-container`

## Build
- Navigate to the home directory of the docker conatiner `cd ~`
- Source the "build.sh" script `source build.sh`
- Run the ros node `ros2 run sensor_fusion hello_world`


