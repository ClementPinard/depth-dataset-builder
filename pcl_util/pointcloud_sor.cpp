#include <glog/logging.h>
#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/ply_io.h>

template<typename PointT>
int sor(std::string input_path, std::string output_path, int knn, int std)
{
  // Load point cloud with normals.
  LOG(INFO) << "Loading point cloud ...";
  typename pcl::PointCloud<PointT>::Ptr input(
      new pcl::PointCloud<PointT>());
  if (pcl::io::loadPLYFile(input_path, *input) < 0) {
    return EXIT_FAILURE;
  }
  LOG(INFO) << "point cloud loaded...";

  // Filter to get inlier cloud, store in filtered_cloud.
  typename pcl::PointCloud<PointT>::Ptr ouptut (new pcl::PointCloud<PointT>);
  typename pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud(input);
  sor.setMeanK(knn);
  sor.setStddevMulThresh(std);
  sor.filter(*ouptut);

  if (!output_path.empty())
    pcl::io::savePLYFileBinary(output_path, *ouptut);

 return (0);
}

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
    LOG(INFO) << "Usage: " << argv[0] << " --input <file.ply> "
              << "--knn <int> --std <int> --output <file.ply>)";
    return EXIT_FAILURE;
  }
  
  std::string input_path;
  pcl::console::parse_argument(argc, argv, "--input", input_path);
  std::string output_path;
  pcl::console::parse_argument(argc, argv, "--output", output_path);
  float knn = 1;
  pcl::console::parse_argument(argc, argv, "--knn", knn);
  float std = 1;
  pcl::console::parse_argument(argc, argv, "--std", std);
  bool with_normals = pcl::console::find_switch(argc, argv, "-n");

  if (output_path.empty()){
    LOG(ERROR) << "No output path was given";
    LOG(INFO) << "Usage: " << argv[0] << " --georef <file.ply> --lidar <file.ply> "
              << "--max_distance <int> --output_matrix <file.txt> (--output_cloud <file.ply>)";
    return EXIT_FAILURE;
  }
  
  if(with_normals){
    return sor<pcl::PointNormal>(input_path, output_path, knn, std);
  }else{
    return sor<pcl::PointXYZ>(input_path, output_path, knn, std);
  }
  
}