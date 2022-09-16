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

# Download the weights
cd colcon_workspace/src/darknet_ros/darknet_ros/yolo_network_config/weights/
wget http://pjreddie.com/media/files/yolov2.weights
wget http://pjreddie.com/media/files/yolov2-tiny.weights
wget http://pjreddie.com/media/files/yolov2-voc.weights
wget http://pjreddie.com/media/files/yolov2-tiny-voc.weights
wget http://pjreddie.com/media/files/yolov3-tiny.weights
wget http://pjreddie.com/media/files/yolov3.weights


# Source darknet
source /colcon_workspace/install/local_setup.bash