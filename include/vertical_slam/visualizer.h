#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include <vector>

struct cloud_point_index_idx {
  unsigned int idx;
  unsigned int cloud_point_index;

  cloud_point_index_idx(int idx_, unsigned int cloud_point_index_) : idx(idx_), cloud_point_index(cloud_point_index_) {}
  bool operator<(const cloud_point_index_idx& p) const { return (idx < p.idx); }
};

struct voxel_index_idx {
  unsigned int idx;
  unsigned int voxel_index;

  voxel_index_idx(int idx_, unsigned int voxel_index_) : idx(idx_), voxel_index(voxel_index_) {}
  bool operator<(const voxel_index_idx& p) const { return (idx < p.idx); }
};

class Visualizer {
 private:
  ros::NodeHandle n_;
  ros::Publisher point_pub_;
  ros::Publisher line_pub_;
  ros::Publisher line_density_pub_;
  ros::Subscriber pc_sub_;
  std::vector<cloud_point_index_idx> index_vector;  // Storage for mapping leaf and pointcloud indexes
  std::vector<voxel_index_idx> v_index_vector;      // Storage for mapping leaf and pointcloud indexes
  double voxel_size = 0.2;
  unsigned int min_points_per_voxel = 2;

 public:
  Visualizer();

  void ResetMarker();

  void SetIndexVector(pcl::PointCloud<pcl::PointXYZ>& input, double voxel_size);

  void Voxelize(pcl::PointCloud<pcl::PointXYZ>& input, pcl::PointCloud<pcl::PointXYZ>& output, double voxel_size,
                unsigned int min_points_per_voxel);

  void ExtractLine(pcl::PointCloud<pcl::PointXYZ>& v_input,
                   std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>>& output);

  void HSVtoRGB(int h, int s, int v, int& r, int& g, int& b);

  void VisualizeVoxel(pcl::PointCloud<pcl::PointXYZ>& input, double voxel_size);

  void VisualizeLine(std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>>& lines);

  void VisualizeLineDensity(std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>>& lines, double voxel_size);

  void GrabPC(const sensor_msgs::PointCloud2& input);
};