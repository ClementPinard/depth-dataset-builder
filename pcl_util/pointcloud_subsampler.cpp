#include <glog/logging.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/console/parse.h>
#include "pointcloud_subsampler.h"

int
main (int argc, char** argv)
{
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
  
  // Parse arguments.
  int dummy;
  if (argc <= 1 ||
      pcl::console::parse_argument(argc, argv, "-h", dummy) >= 0 ||
      pcl::console::parse_argument(argc, argv, "--help", dummy) >= 0) {
    LOG(INFO) << "Usage: " << argv[0] << " --point_cloud_path <file.ply> --resolution <m> --out_mesh <file.ply>";
    return EXIT_FAILURE;
  }
  
  std::string point_cloud_path;
  pcl::console::parse_argument(argc, argv, "--point_cloud_path", point_cloud_path);
  float resolution = 0.2; //20cm resolution
  pcl::console::parse_argument(argc, argv, "--resolution", resolution);
  std::string output_path;
  pcl::console::parse_argument(argc, argv, "--output", output_path);
  
  // Load point cloud with normals.
  LOG(INFO) << "Loading point cloud ...";
  pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  if (pcl::io::loadPLYFile(point_cloud_path, *point_cloud) < 0) {
    return EXIT_FAILURE;
  }

  LOG(INFO) << "Subsampling to have a mean distance between points of " << resolution << " m";
  point_cloud = filter<pcl::PointXYZ>(point_cloud, resolution);

  pcl::io::savePLYFileBinary (output_path, *point_cloud);

  return (0);
}