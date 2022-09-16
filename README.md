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
- Navigate to the home directory of the docker container `cd ~`
- Source the "build.sh" script `source /rosEnv/build.sh`
- Run the "darknet.sh" script `./scripts/darknet.sh`
- Source darknet_ros's local setup `source /colcon_workspace/install/local_setup.bash`
- Run the ros node `ros2 run sensor_fusion hello_world`
- To run darknet, run `ros2 launch darknet_ros darknet_ros.launch.py`

## Git best-practices
### Merging to main
1. `git switch main`
2. `git pull`
3. `git switch {yourBranch}`
4. `git rebase -i main`
5. Squash down to one commit by changing all commits to `fastforward` except for the first.
6. Resolve any conflicts.
7. `git push -f` -f is needed because of the rebase. Only use this on your own branch.
8. Go to the "Pull request" tab on GitHub and click "New pull request" to create a PR of your branch onto main.
9. In the "Reviewers" sidebar, request a review from everyone on the team.
10. Make any changes and get approval.
11. In you PR on GitHub, select "Squash and merge" (You may have to look for this in a dropdown menu)

## Other
- To edit files from git in VSCode instead of VIM run `git config --global core.editor "code --wait"`