# StratoMiners
Field session (CSCI-370) project for Stratom

## Getting Started

### Install
- Use Ubuntu 22.04 LTS
- Install docker as documented [here](https://docs.docker.com/desktop/install/linux-install/)
- run `sudo usermod -aG docker {username}` and restart Ubuntu. This sets permissions so `sudo` doesn't have to be used in docker commands
- run `git clone --recursive https://github.com/EthanCranston/StratoMiners.git` to pull the repo.
Then, either source env_setup.sh, or
- run `./env_setup.sh`
- run `docker compose up -d` to start container

### Build
- Navigate to the workspace of the docker container `cd ~`
- Build all packages with colcon `colcon build --symlink-install`
- Source the "build.sh" script to source all packages `source build.sh`

### Usage
- Start the project using the alias in the "build.sh" script `start_node`
- From there, a configurable RViz interface will appear displaying all nodes

### Configuration
All nodes are configurable using YAML files in similar formats to those currently in use. These are found in the following config folders:
- `/rosEnv/src/darknet_ros/darknet_ros/config`
- `/rosEnv/src/gb_visual_detection_3d/darknet_ros_3d/config`
- `/rosEnv/src/lidar_cv/config`
- `/rosEnv/src/sensor_fusion/config`

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

### Updating submodules
1. Navigate to the submodule folder, commit all changes, and push it to the submodule repository
2. Then, navigate to the StratoMiners folder
3. Update the submodules using `git submodule update --remote --init --recursive`
4. Make a new commit to the StratoMiners repository once your submodules are updated so that they are refreshed remotely.

### Other
- To edit files from git in VSCode instead of VIM run `git config --global core.editor "code --wait"`
- See [Darknet](docs/Darknet.md) for more on using Darknet.

## References

Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.

    @article{opencv_library,
        author = {Bradski, G.},
        citeulike-article-id = {2236121},
        journal = {Dr. Dobb's Journal of Software Tools},
        keywords = {bibtex-import},
        posted-at = {2008-01-15 19:21:54},
        priority = {4},
        title = {{The OpenCV Library}},
        year = {2000}
    }

S. Macenski, T. Foote, B. Gerkey, C. Lalancette, W. Woodall, “Robot Operating System 2: Design, architecture, and uses in the wild,” Science Robotics vol. 7, May 2022.

    @article{
        doi:10.1126/scirobotics.abm6074,
        author = {Steven Macenski  and Tully Foote  and Brian Gerkey  and Chris Lalancette  and William Woodall },
        title = {Robot Operating System 2: Design, architecture, and uses in the wild},
        journal = {Science Robotics},
        volume = {7},
        number = {66},
        pages = {eabm6074},
        year = {2022},
        doi = {10.1126/scirobotics.abm6074},
        URL = {https://www.science.org/doi/abs/10.1126/scirobotics.abm6074}
    }

Redmon, J. (2013–2016). Darknet: Open Source Neural Networks in C. .

    @misc{darknet13,
        author =   {Redmon, J},
        title =    {Darknet: Open Source Neural Networks in C},
        howpublished = {\url{http://pjreddie.com/darknet/}},
        year = {2013--2016}
    }

Wang, C.Y., Bochkovskiy, A., & Liao, H.Y. (2022). YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv preprint arXiv:2207.02696.

    @article{wang2022yolov7,
        title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
        author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
        journal={arXiv preprint arXiv:2207.02696},
        year={2022}
    }

