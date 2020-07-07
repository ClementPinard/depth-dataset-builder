cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)

project(Parrot_photogrammetry_PCLutils)

find_package(PCL REQUIRED)
find_package(COLMAP REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${COLMAP_INCLUDE_DIRS})
link_directories(${COLMAP_LINK_DIRS})

add_executable (PointCloudSubsampler pointcloud_subsampler.cpp)
target_link_libraries (PointCloudSubsampler ${PCL_LIBRARIES} glog)

add_executable (MeshTriangulator mesh_triangulator.cpp)
target_link_libraries (MeshTriangulator ${PCL_LIBRARIES} glog)

add_executable (CloudRegistrator cloud_registrator.cpp)
target_link_libraries(CloudRegistrator ${PCL_LIBRARIES} glog)

add_executable (CloudSOR pointcloud_sor.cpp)
target_link_libraries(CloudSOR ${PCL_LIBRARIES} glog)

add_executable (NormalsTransfer normals_transfer.cpp)
target_link_libraries(NormalsTransfer ${PCL_LIBRARIES} glog)

add_executable (CreateVisFile create_vis_file.cpp)
target_link_libraries(CreateVisFile ${PCL_LIBRARIES} glog ${COLMAP_LIBRARIES})
