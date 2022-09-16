# First, create the workspace
cd ../..
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

# Build and source darknet
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source /colcon_workspace/install/local_setup.bash