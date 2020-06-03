#!/bin/bash

cd pcl_util \
   && mkdir build \
   && cd build \
   && cmake .. \
   && make -j8