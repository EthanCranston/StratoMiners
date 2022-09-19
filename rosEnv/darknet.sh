# First, create the workspace
cd /
sudo apt-get update -y
mkdir colcon_workspace
cd colcon_workspace
mkdir src
cd src

# Pull all darknet images
git clone https://github.com/Ar-Ray-code/darknet_ros.git -b master --recursive
bash /colcon_workspace/src/darknet_ros/rm_darknet_CMakeLists.sh

# Replace CMakeLists.txt to bypass CUDA requirement
rm -rf /colcon_workspace/src/darknet_ros/darknet_ros/CMakeLists.txt
cp -fr /rosEnv/darknet/CMakeLists.txt /colcon_workspace/src/darknet_ros/darknet_ros

# Build darknet
cd /colcon_workspace
colcon build --symlink-install

# Replace config and launch files
cp -fr /rosEnv/darknet/ros-stratominers.yaml /colcon_workspace/install/darknet_ros/share/darknet_ros/config
cp -fr /rosEnv/darknet/yolov7-tiny-stratominers.yaml /colcon_workspace/install/darknet_ros/share/darknet_ros/config
cp -fr /rosEnv/darknet/stratominers.launch.py /colcon_workspace/install/darknet_ros/share/darknet_ros/launch