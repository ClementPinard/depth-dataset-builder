#!/bin/bash

# This scrip helps you install the necessary tools to construct a depth enabled dataset with Anafi videos
# Note that CUDA needs to be already installed

sudo apt update
sudo apt install -y git \
    repo \
    cmake \
    build-essential \
    pkg-config \
    libboost-all-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libvtk6-dev \
    libflann-dev \
    cmake libgmp-dev \
    libgoogle-glog-dev \
    qt5-default \
    libqwt-qt5-dev \
    libpcl-dev \
    libproj-dev \
    libcgal-qt5-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    zlib1g-dev \
    libglfw3-dev \
    libsdl2-dev rsync

git clone https://github.com/laurentkneip/opengv.git
cd opengv \
  && mkdir build \
  && cd build \
  && cmake ..
  && make -j8
sudo make install
cd ../../

git clone https://github.com/opencv/opencv.git
cd opencv \
  && git checkout 4.1.2 \
  && mkdir build \
  && cd build \
  && cmake -D WITH_CUDA=OFF .. \
  && make -j8
sudo make install
cd ../../

git clone https://github.com/ETH3D/dataset-pipeline
cd dataset-pipeline \
  && mkdir -p build \
  && cd build \
  && cmake .. \
  && make -j8 \
  && cd ../../

git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver \
  && git checkout $(git describe --tags) \
  && mkdir build \
  && cd build
  && cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
  && make -j8
sudo make install
cd ../../

git clone https://github.com/colmap/colmap.git
cd colmap \
  && git checkout dev \
  && mkdir build
  && cd build \
  && cmake .. \
  && make -j8
sudo make install
cd ../../

mkdir -p groundsdk \
  && cd groundsdk \
  && repo init -u https://github.com/Parrot-Developers/groundsdk-manifest -m release.xml \
  && repo sync \
  && ./build.sh -p pdraw-linux -t build -j8
cd ../

pip install -r requirements.txt

./build_pcl.sh