# First, create the workspace
cd /
mkdir colcon_workspace
cd colcon_workspace
mkdir src
cd src

# Pull all darknet images
git clone https://github.com/mzdavid08/darknet_ros.git -b master --recursive

# Build darknet
cd /colcon_workspace
colcon build --symlink-install

# Return to home directory and install sensor_fusion
cd /rosEnv
colcon build --packages-skip darknet_ros