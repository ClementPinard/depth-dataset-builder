#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/filters/extract_indices.h>

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr filter(typename pcl::PointCloud<PointT>::Ptr input, float resolution) {
  
  typename pcl::octree::OctreePointCloudPointVector<PointT> octree (100*resolution);
  octree.setInputCloud(input);
  octree.addPointsFromInputCloud();

  typename pcl::ExtractIndices<PointT> extract;

  pcl::PointIndices::Ptr to_process (new pcl::PointIndices ());
  typename pcl::PointCloud<PointT>::Ptr input_subcloud (new pcl::PointCloud<PointT>);
  typename pcl::PointCloud<PointT>::Ptr final_cloud (new pcl::PointCloud<PointT>);
  typename pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
  pcl::VoxelGrid<PointT> vox;
  vox.setLeafSize (resolution, resolution, resolution);

  for (auto it = octree.leaf_begin(); it != octree.leaf_end(); ++it) {
    pcl::octree::OctreeContainerPointIndices& container = it.getLeafContainer();

    pcl::IndicesPtr indexVector(new std::vector<int>);
    container.getPointIndices(*indexVector);
    extract.setInputCloud (input);
    extract.setIndices (indexVector);
    extract.filter (*input_subcloud);

    vox.setInputCloud (input_subcloud);
    vox.filter (*cloud_filtered);
    *final_cloud += *cloud_filtered;
  }
  return final_cloud;
}