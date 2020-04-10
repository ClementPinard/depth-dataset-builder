cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(Parrot_photogrammetry_PCLutils)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (PointCloudSubsampler pointcloud_subsampler.cpp)
target_link_libraries (PointCloudSubsampler ${PCL_LIBRARIES} glog)

add_executable (MeshTriangulator mesh_triangulator.cpp)
target_link_libraries (MeshTriangulator ${PCL_LIBRARIES} glog)

add_executable (CloudRegistrator cloud_registrator.cpp)
target_link_libraries(CloudRegistrator ${PCL_LIBRARIES} glog)

add_executable (CloudSOR pointcloud_sor.cpp)
target_link_libraries(CloudSOR ${PCL_LIBRARIES} glog)
