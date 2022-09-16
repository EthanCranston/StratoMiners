# First, create the workspace
cd /
sudo apt-get update -y
mkdir colcon_workspace
cd colcon_workspace
mkdir src
cd src

# Pull all darknet images
git clone https://github.com/leggedrobotics/darknet_ros.git -b foxy
cd darknet_ros
git clone https://github.com/leggedrobotics/darknet.git -b yolov3
cd ../..

# Build darknet
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Copy launch file
cp -fr /rosEnv/scripts/darknet_ros_stratominers.launch.py /colcon_workspace/install/darknet_ros/share/darknet_ros/launch