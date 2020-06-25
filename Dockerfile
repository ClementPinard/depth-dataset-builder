FROM nvidia/cudagl:10.2-devel-ubuntu18.04

WORKDIR /parrot-photogrammetry
COPY . /parrot-photogrammetry
ARG DEBIAN_FRONTEND=noninteractive


RUN apt update \
  && apt install -y git \
    repo \
    python3 \
    python3-pip \
    python \
    gosu \
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
    libqwt-qt5-dev \
    libpcl-dev \
    libproj-dev \
    libcgal-qt5-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    zlib1g-dev \
    libglfw3-dev \
    libsdl2-dev rsync

RUN git clone https://github.com/laurentkneip/opengv.git
RUN cd opengv \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j8 \
  && make install \
  && cd ../../

RUN git clone https://github.com/opencv/opencv.git
RUN cd opencv \
  && git checkout 4.1.2 \
  && mkdir build \
  && cd build \
  && cmake -D WITH_CUDA=OFF .. \
  && make -j8 \
  && make install \
  && cd ../../

RUN git clone https://github.com/ETH3D/dataset-pipeline
RUN cd dataset-pipeline \
  && mkdir -p build \
  && cd build \
  && cmake .. \
  && make -j8 \
  && cd ../../

RUN git clone https://ceres-solver.googlesource.com/ceres-solver
RUN cd ceres-solver \
  && git checkout $(git describe --tags) \
  && mkdir build \
  && cd build \
  && cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
  && make -j8 \
  && make install \
  && cd ../../

RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap \
  && git checkout dev \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j8 \
  && make install \
  && cd ../../

RUN mkdir -p groundsdk \
  && cd groundsdk \
  && repo init -u https://github.com/Parrot-Developers/groundsdk-manifest -m release.xml \
  && repo sync
RUN mkdir -p /.config && chmod 777 -R /.config && chmod 777 -R groundsdk/ && cd groundsdk/ && yes "y" | gosu 1000:1 ./build.sh -p pdraw-linux -t build -j8 \
  && cd ../

RUN pip3 install -r requirements.txt

RUN /parrot-photogrammetry/build_pcl_util.sh
