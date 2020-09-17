#!/bin/bash

# This script helps you install the necessary tools to construct a depth enabled dataset with Anafi videos
# Note that CUDA and Anaconda need to be already installed
# Also note that for repo to work, git needs to be parametrized with email and name.


# This command makes sure that the .so files pointed by the cmake commands are the right ones
# Alternatively, you can add conda to the cmake folders with the following command :
# CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# Note that the alternative will trigger erros if you move or delete the anaconda folder.

eval "$(conda shell.bash hook)"
conda deactivate

sudo apt update
sudo apt install -y git \
    curl \
    cmake \
    ffmpeg \
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
    libflann-dev \
    cmake libgmp-dev \
    libgoogle-glog-dev \
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
  && mkdir -p build \
  && cd build \
  && cmake .. \
  && make -j8
sudo make install
cd ../../

git clone https://github.com/opencv/opencv.git
cd opencv \
  && git checkout 4.1.2 \
  && mkdir -p build \
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
  && mkdir -p build \
  && cd build \
  && cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
  && make -j8
sudo make install
cd ../../

git clone https://github.com/colmap/colmap.git
cd colmap \
  && git checkout dev \
  && mkdir -p build \
  && cd build \
  && cmake .. \
  && make -j8
sudo make install
cd ../../

mkdir -p ~/.bin
PATH="${HOME}/.bin:${PATH}"
curl https://storage.googleapis.com/git-repo-downloads/repo > ~/.bin/repo
chmod a+rx ~/.bin/repo

conda activate
mkdir -p groundsdk \
  && cd groundsdk \
  && repo init -u https://github.com/Parrot-Developers/groundsdk-manifest -m release.xml \
  && repo sync \
  && ./build.sh -p pdraw-linux -t build -j8
cd ../

pip install -r requirements.txt

./build_pcl_util.sh