#include <geometry_msgs/Point.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include <vector>

#include "vertical_slam/HeightGrid.h"

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

class FeatureExtractor {
 private:
  ros::NodeHandle n_;
  ros::Publisher point_pub_;
  ros::Publisher line_pub_;
  ros::Publisher line_density_pub_;
  ros::Publisher height_grid_pub_;
  ros::Publisher height_grid_occ_pub_;
  ros::Subscriber pc_sub_;
  std::vector<cloud_point_index_idx> index_vector;  // Storage for mapping leaf and pointcloud indexes
  std::vector<voxel_index_idx> v_index_vector;      // Storage for mapping leaf and pointcloud indexes
  double voxel_size = 0.2;
  unsigned int min_points_per_voxel = 2;
  HeightGrid first_height_grid;
  HeightGrid second_height_grid;

 public:
  FeatureExtractor() {
    point_pub_ = n_.advertise<visualization_msgs::MarkerArray>("/visualization_marker/point", 1);
    line_pub_ = n_.advertise<visualization_msgs::MarkerArray>("/visualization_marker/line", 1);
    line_density_pub_ = n_.advertise<visualization_msgs::MarkerArray>("/visualization_marker/line_density", 1);
    height_grid_pub_ = n_.advertise<visualization_msgs::MarkerArray>("/visualization_marker/height_grid", 1);
    height_grid_occ_pub_ = n_.advertise<nav_msgs::OccupancyGrid>("/visualization_marker/height_grid_occ", 1);
    pc_sub_ = n_.subscribe("/kitti/velo/pointcloud", 1, &FeatureExtractor::GrabPC, this);
  }

  void GrabPC(const sensor_msgs::PointCloud2& input) {
    // ResetMarker();
    // convet input to xyz
    pcl::PointCloud<pcl::PointXYZ>::Ptr ptr_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(input, *ptr_cloud);

    // voxelization
    pcl::PointCloud<pcl::PointXYZ>::Ptr ptr_voxelized(new pcl::PointCloud<pcl::PointXYZ>);
    SetIndexVector(*ptr_cloud, voxel_size);
    Voxelize(*ptr_cloud, *ptr_voxelized, voxel_size, min_points_per_voxel);

    // line extraction
    std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>> lines;
    ExtractLine(*ptr_voxelized, lines);

    // visualization
    // VisualizeVoxel(*ptr_voxelized, voxel_size);
    // VisualizeLine(lines);
    // VisualizeLineDensity(lines, voxel_size);

    if (first_height_grid.GetCells().size() == 0) {
      first_height_grid = GetHeightGridFromLines(lines, voxel_size);
      VisualizeHeightGrid(first_height_grid, 0);
      // RunTestICP();
    } else if (second_height_grid.GetCells().size() == 0) {
      second_height_grid = GetHeightGridFromLines(lines, voxel_size);
      VisualizeHeightGrid(second_height_grid, 1);
      RunICP(first_height_grid, second_height_grid);
    }
    // VisualizeHeightGridInOccupancyGrid(first_height_grid);
  }

  void SetIndexVector(pcl::PointCloud<pcl::PointXYZ>& input, double voxel_size);

  void Voxelize(pcl::PointCloud<pcl::PointXYZ>& input, pcl::PointCloud<pcl::PointXYZ>& output, double voxel_size,
                unsigned int min_points_per_voxel);

  void ExtractLine(pcl::PointCloud<pcl::PointXYZ>& v_input,
                   std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>>& output);

  ////////////////////////////////////
  ////////// HeightGrid ICP //////////
  HeightGrid GetHeightGridFromLines(std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>>& lines, double voxel_size);
  Point GetCentroid(HeightGrid& height_grid);
  Point GetCentroidWithoutHeight(HeightGrid& height_grid);
  void DemeanHeightGrid(HeightGrid& height_grid_in, HeightGrid& height_grid_out, Point centroid);
  void RunICP(HeightGrid& first_height_grid, HeightGrid& second_height_grid);
  void RunTestICP();
  Eigen::Matrix3d FindAlignment(HeightGrid& P, HeightGrid& Y);

  ///////////////////////////////////
  ////////// Visualization //////////
  void ResetMarker();
  void VisualizeVoxel(pcl::PointCloud<pcl::PointXYZ>& input, double voxel_size);
  void VisualizeLine(std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>>& lines);
  void VisualizeLineDensity(std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>>& lines, double voxel_size);
  void VisualizeHeightGrid(HeightGrid& height_grid, int color);  // 0: red, 1: green
  void VisualizeLineBetweenMatchingPoints(HeightGrid HG1, HeightGrid HG2);
  void VisualizeCentroid(Eigen::Vector2d centroid, time_t timestamp, int color);  // 0: red, 1: green, 2; blue
  void VisualizeHeightGridInOccupancyGrid(HeightGrid& height_grid);
  ///////////////////////////////////
  ///////////////////////////////////

  void HSVtoRGB(int h, int s, int v, int& r, int& g, int& b);
};
