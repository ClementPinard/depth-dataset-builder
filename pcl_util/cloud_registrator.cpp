#include <glog/logging.h>
#include <iostream>
#include <fstream>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_svd_scale.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/ply_io.h>


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
  
  std::string georef_path;
  pcl::console::parse_argument(argc, argv, "--georef", georef_path);
  std::string lidar_path;
  pcl::console::parse_argument(argc, argv, "--lidar", lidar_path);
  float max_distance = 1; //1m max distance
  pcl::console::parse_argument(argc, argv, "--max_distance", max_distance);
  std::string output_matrix_path;
  pcl::console::parse_argument(argc, argv, "--output_matrix", output_matrix_path);
  std::string output_cloud_path;
  pcl::console::parse_argument(argc, argv, "--output_cloud", output_cloud_path);

  if (output_matrix_path.empty() && output_cloud_path.empty()){
    LOG(ERROR) << "No output path was given";
    LOG(INFO) << "Usage: " << argv[0] << " --georef <file.ply> --lidar <file.ply> "
              << "--max_distance <int> --output_matrix <file.txt> (--output_cloud <file.ply>)";
    return EXIT_FAILURE;
  }
  
  // Load point cloud with normals.
  LOG(INFO) << "Loading point clouds ...";
  pcl::PointCloud<pcl::PointXYZ>::Ptr georef(
      new pcl::PointCloud<pcl::PointXYZ>());
  if (pcl::io::loadPLYFile(georef_path, *georef) < 0) {
    return EXIT_FAILURE;
  }

  pcl::PointCloud<pcl::PointNormal>::Ptr lidar(
      new pcl::PointCloud<pcl::PointNormal>());
  if (pcl::io::loadPLYFile(lidar_path, *lidar) < 0) {
    return EXIT_FAILURE;
  }

  LOG(INFO) << "point clouds loaded...";

  // Filter to get inlier cloud, store in filtered_cloud.
  //pcl::PointCloud<pcl::PointXYZ>::Ptr geroef_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  //pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  //sor.setInputCloud(geroef);
  //sor.setMeanK(6);
  //sor.setStddevMulThresh(0.1);
  //sor.filter(*geroef_filtered);


  // Normal estimation*
  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (georef);
  n.setInputCloud (georef);
  n.setSearchMethod (tree);
  n.setKSearch (6);
  n.compute (*normals);
  pcl::PointCloud<pcl::PointNormal>::Ptr geroef_normals (new pcl::PointCloud<pcl::PointNormal>);
  pcl::concatenateFields (*georef, *normals, *geroef_normals);

  // pcl::io::savePLYFile("test_normals.ply", *geroef_normals);

  pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> icp;
  pcl::registration::TransformationEstimationSVDScale<pcl::PointNormal, pcl::PointNormal>::Ptr est;
  est.reset(new pcl::registration::TransformationEstimationSVDScale<pcl::PointNormal, pcl::PointNormal>);
  icp.setTransformationEstimation(est);

  icp.setMaxCorrespondenceDistance (max_distance);
  icp.setTransformationEpsilon(0.0001);
  icp.setMaximumIterations(500);
  icp.setEuclideanFitnessEpsilon(0.0001);
  icp.setInputSource(geroef_normals);
  icp.setInputTarget(lidar);
  
  pcl::PointCloud<pcl::PointNormal> Final;
  icp.align(Final);
  Eigen::Matrix4f transform = icp.getFinalTransformation();
  pcl::PointCloud<pcl::PointNormal>::Ptr lidar_aligned(
      new pcl::PointCloud<pcl::PointNormal>());
  pcl::transformPointCloud (*lidar, *lidar_aligned, transform);
  
  std::ofstream output_file;
  output_file.open(output_matrix_path);
  output_file << transform << std::endl;
  output_file.close();

  if (!output_cloud_path.empty())
    pcl::io::savePLYFileBinary(output_cloud_path, *lidar_aligned);

 return (0);
}