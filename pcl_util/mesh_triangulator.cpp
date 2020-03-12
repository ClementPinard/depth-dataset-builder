// Copyright 2020 ENSTA Paris, Clément Pinard
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/gp3.h>
#include "pointcloud_subsampler.h"

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
  
  // Parse arguments.
  int dummy;
  if (argc <= 1 ||
      pcl::console::parse_argument(argc, argv, "-h", dummy) >= 0 ||
      pcl::console::parse_argument(argc, argv, "--help", dummy) >= 0) {
    LOG(INFO) << "Usage: " << argv[0] << " --point_normal_cloud_path <file.ply> --resolution <m> --out_mesh <file.ply>";
    return EXIT_FAILURE;
  }
  
  std::string point_normal_cloud_path;
  pcl::console::parse_argument(argc, argv, "--point_normal_cloud_path", point_normal_cloud_path);
  float resolution = 20; //20cm resolution
  pcl::console::parse_argument(argc, argv, "--resolution", resolution);
  std::string mesh_output;
  pcl::console::parse_argument(argc, argv, "--out_mesh", mesh_output);
  
  // Load point cloud with normals.
  LOG(INFO) << "Loading point cloud ...";
  pcl::PointCloud<pcl::PointNormal>::Ptr point_normal_cloud(
      new pcl::PointCloud<pcl::PointNormal>());
  if (pcl::io::loadPLYFile(point_normal_cloud_path, *point_normal_cloud) < 0) {
    return EXIT_FAILURE;
  }

  LOG(INFO) << "Subsampling to have a mean distance between points of " << resolution << " m";
  point_normal_cloud = filter<pcl::PointNormal>(point_normal_cloud, resolution);
  
  LOG(INFO) << "Beginning triangulation";
  // Create search tree*
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointNormal>);
  tree->setInputCloud (point_normal_cloud);

  pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
  pcl::PolygonMesh triangles;

  // Set the maximum distance between connected points (maximum edge length)
  gp3.setSearchRadius (resolution * 2);

  // Set typical values for the parameters
  gp3.setMu (2.5);
  gp3.setMaximumNearestNeighbors (100);
  gp3.setMaximumSurfaceAngle(M_PI);
  gp3.setMinimumAngle(0);
  gp3.setMaximumAngle(2 * M_PI);
  gp3.setNormalConsistency(false);

  // Get result
  gp3.setInputCloud (point_normal_cloud);
  gp3.setSearchMethod (tree);
  gp3.reconstruct (triangles);

  LOG(INFO) << "Done.";

  // Additional vertex information
  if (!mesh_output.empty())
    pcl::io::savePLYFileBinary (mesh_output, triangles);

  return EXIT_SUCCESS;
}
