# StratoMiners
Field session (CSCI-370) project for Stratom

## Getting Started
- Use Ubuntu 22.04 LTS
- Install docker as documented [here](https://docs.docker.com/desktop/install/linux-install/)
- run `sudo usermod -aG docker {username}` and restart Ubuntu. This sets permissions so `sudo` doesn't have to be used in docker commands
- run `git clone https://github.com/EthanCranston/StratoMiners.git` to pull the repo.
Then, either source env_setup.sh, or
- run `./env_setup.sh`
- run `docker compose up -d` to start container

## Build
- Navigate to the home directory of the docker conatiner `cd ~`
- Source the "build.sh" script `source build.sh`
- Run the ros node `ros2 run sensor_fusion hello_world`


## Git best-practices
### Mergeing to main
1. `git switch main`
2. `git pull`
3. `git switch {yourBranch}`
4. `git rebase -i main`
5. Squash down to one commit by changing all commits to `fastforward` except for the first
6. Resolve any conflicts
7. `git switch main`
8. `git merge {yourBranch}`
## Other
- To edit files from git in VSCode instead of VIM run `git config --global core.editor "code --wait"`