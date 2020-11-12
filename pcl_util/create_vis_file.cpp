#include <glog/logging.h>
#include <iostream>
#include <fstream>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/ply_io.h>

#include <colmap/mvs/fusion.h>
#include <colmap/util/endian.h>

#include "pointcloud_subsampler.h"


int main (int argc, char** argv)
{
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
  
  // Parse arguments.
  int dummy;
  if (argc <= 1 ||
      pcl::console::parse_argument(argc, argv, "-h", dummy) >= 0 ||
      pcl::console::parse_argument(argc, argv, "--help", dummy) >= 0) {
    LOG(INFO) << "Usage: " << argv[0] << " --georef_dense <file.ply> --lidar <file.ply> "
              << "--georef_matrix <file.txt> --resolution <float> (--output_cloud <file.ply>)";
    return EXIT_FAILURE;
  }
  
  std::string georef_dense_path;
  pcl::console::parse_argument(argc, argv, "--georef_dense", georef_dense_path);
  std::string lidar_path;
  pcl::console::parse_argument(argc, argv, "--lidar", lidar_path);
  std::string output_cloud_path;
  pcl::console::parse_argument(argc, argv, "--output_cloud", output_cloud_path);
  float resolution = 0.2; //20cm resolution
  pcl::console::parse_argument(argc, argv, "--resolution", resolution);
  float max_distance = 10;
  pcl::console::parse_argument(argc, argv, "--max_distance", max_distance);

  if (output_cloud_path.empty()){
    LOG(ERROR) << "No output path was given";
    LOG(INFO) << "Usage: " << argv[0] << " --georef_dense <file.ply> --lidar <file.ply> "
              << "--output_cloud <output.ply>";
    return EXIT_FAILURE;
  }
  
  // Load point cloud with normals.
  LOG(INFO) << "Loading point clouds ...";
  pcl::PointCloud<pcl::PointNormal>::Ptr georef_dense(
      new pcl::PointCloud<pcl::PointNormal>());
  if (pcl::io::loadPLYFile(georef_dense_path, *georef_dense) < 0) {
    return EXIT_FAILURE;
  }

  pcl::PointCloud<pcl::PointNormal>::Ptr lidar(
      new pcl::PointCloud<pcl::PointNormal>());
  if (pcl::io::loadPLYFile(lidar_path, *lidar) < 0) {
    return EXIT_FAILURE;
  }
  LOG(INFO) << "point clouds loaded";

  LOG(INFO) << "Subsampling Lidar point cloud to have a mean distance between points of " << resolution << " m";
  lidar = filter<pcl::PointNormal>(lidar, resolution);

  LOG(INFO) << "Loading georef_dense vis file...";
  const std::string input_vis_path = georef_dense_path + ".vis";
  std::fstream input_vis_file(input_vis_path, std::ios::in | std::ios::binary);
  CHECK(input_vis_file.is_open()) << input_vis_path;

  const size_t vis_num_points = colmap::ReadBinaryLittleEndian<uint64_t>(&input_vis_file);
  CHECK_EQ(vis_num_points, georef_dense->size());

  std::vector<std::vector<int>> input_vis_points;
  input_vis_points.reserve(georef_dense->size());
  for (auto it=georef_dense->begin(); it!=georef_dense->end(); it++) {
    std::vector<int> image_idx;
    int num_visible_images =
        colmap::ReadBinaryLittleEndian<int>(&input_vis_file);
    image_idx.reserve(num_visible_images);
    for (uint32_t i = 0; i < num_visible_images; ++i) {
      image_idx.push_back(colmap::ReadBinaryLittleEndian<uint32_t>(&input_vis_file));
    }
    input_vis_points.push_back(image_idx);
  }

  LOG(INFO) << "visible images ids ready to be transferred";

  pcl::KdTree<pcl::PointNormal>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointNormal>);
  tree->setInputCloud(georef_dense);
  std::vector<int> nn_indices (1);
  std::vector<float> nn_dists (1);

  std::vector<std::vector<int>> output_vis_points;
  output_vis_points.reserve(lidar->size());

  for(auto it = lidar->begin(); it != lidar->end(); it++){
    tree->nearestKSearch(*it, 1, nn_indices, nn_dists);
    if(nn_dists[0] <= max_distance){
      std::vector<int> image_idx = input_vis_points.at(nn_indices[0]);
      output_vis_points.push_back(image_idx);
    }else{
      output_vis_points.push_back(std::vector<int>)
    }
  }

  
  if (!output_cloud_path.empty()) {
    // Note : Instead of using pcl::savePLYFileBinary function,
    // we use a PLYwriter so that can set the camera parameter to false,
    // to not add its element in the header, because COLMAP doesn't like
    // PLY files with unknown headers.
    pcl::PLYWriter writer;
    const bool binary=true, use_camera=false;
    writer.write<pcl::PointNormal>(output_cloud_path, *lidar, binary, use_camera);
    const std::string output_vis_path = output_cloud_path + ".vis";
    colmap::mvs::WritePointsVisibility(output_vis_path, output_vis_points);
  }

 return (0);
}