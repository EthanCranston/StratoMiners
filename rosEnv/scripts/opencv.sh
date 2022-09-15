# Installs OpenCV in root
# This process is outside the dockerfile to save time
cd ../..
mkdir opencv
cd opencv
sudo apt update && sudo apt install -y cmake g++ wget unzip
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip
mkdir -p build && cd build
cmake  ../opencv-4.x
cmake --build .