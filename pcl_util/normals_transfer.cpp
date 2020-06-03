#include <glog/logging.h>
#include <iostream>
#include <fstream>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/ply_io.h>


Eigen::Matrix4f readMatrix(std::string filename)
{
  int cols = 4, rows = 4;
  Eigen::Matrix4f M;

  std::ifstream infile;
  infile.open(filename);
  int row = 0;
  while (! infile.eof() && row < 4)
      {
      std::string line;
      std::getline(infile, line);

      std::stringstream stream(line);
      stream >> M(row, 0) >> M(row, 1) >> M(row, 2) >> M(row, 3);
      row ++;
      }
  infile.close();
  return M;
};


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
    LOG(INFO) << "Usage: " << argv[0] << " --georef <file.ply> --lidar <file.ply> "
              << "--max_distance <int> --output_matrix <file.txt> (--output_cloud <file.ply>)";
    return EXIT_FAILURE;
  }
  
  std::string georef_dense_path;
  pcl::console::parse_argument(argc, argv, "--georef_dense", georef_dense_path);
  std::string lidar_path;
  pcl::console::parse_argument(argc, argv, "--lidar", lidar_path);
  std::string georef_matrix_path;
  pcl::console::parse_argument(argc, argv, "--georef_matrix", georef_matrix_path);
  std::string output_cloud_path;
  pcl::console::parse_argument(argc, argv, "--output_cloud", output_cloud_path);

  if (georef_matrix_path.empty() && output_cloud_path.empty()){
    LOG(ERROR) << "No output path was given";
    LOG(INFO) << "Usage: " << argv[0] << " --georef_dense <file.ply> --lidar <file.ply> "
              << "--georef_matrix <matrix.txt> --output_cloud <output.ply>";
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

  LOG(INFO) << "Loading transformation matrix ...";
  Eigen::Matrix4f M = readMatrix(georef_matrix_path);
  LOG(INFO) << "Matrix loaded";

  // Filter to get inlier cloud, store in filtered_cloud.
  LOG(INFO) << "filtering georef cloud";
  pcl::PointCloud<pcl::PointNormal>::Ptr geroef_filtered (new pcl::PointCloud<pcl::PointNormal>);
  pcl::StatisticalOutlierRemoval<pcl::PointNormal> sor;
  sor.setInputCloud(georef_dense);
  sor.setMeanK(6);
  sor.setStddevMulThresh(0.1);
  sor.filter(*geroef_filtered);

  pcl::PointCloud<pcl::PointNormal>::Ptr georef_aligned(
      new pcl::PointCloud<pcl::PointNormal>());
  pcl::transformPointCloudWithNormals (*geroef_filtered, *georef_aligned, M);

  LOG(INFO) << "Normals ready to be transferred";

  pcl::KdTree<pcl::PointNormal>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointNormal>);
  tree->setInputCloud(georef_aligned);
  std::vector<int> nn_indices (1);
  std::vector<float> nn_dists (1);

  for(auto it = lidar->begin(); it != lidar->end(); it++){
    tree->nearestKSearch(*it, 1, nn_indices, nn_dists);
    auto n1 = georef_aligned->points[nn_indices[0]].getNormalVector3fMap();
    auto n2 = it->getNormalVector3fMap();
    if (n1.dot(n2) < 0){
      n2 *= -1;
    }
  }

  
  if (!output_cloud_path.empty())
    pcl::io::savePLYFileBinary(output_cloud_path, *lidar);

 return (0);
}