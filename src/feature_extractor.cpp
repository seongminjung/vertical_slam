#include "vertical_slam/feature_extractor.h"

#include <geometry_msgs/Point.h>
#include <nav_msgs/OccupancyGrid.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include <vector>

#include "vertical_slam/HeightGrid.h"

void FeatureExtractor::ResetMarker() {
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker deleteall_marker;
  deleteall_marker.ns = "point";
  deleteall_marker.id = 1;
  deleteall_marker.action = visualization_msgs::Marker::DELETEALL;
  marker_array.markers.push_back(deleteall_marker);
  point_pub_.publish(marker_array);
  line_pub_.publish(marker_array);
  line_density_pub_.publish(marker_array);
  height_grid_pub_.publish(marker_array);
  marker_array.markers.clear();
}

void FeatureExtractor::SetIndexVector(pcl::PointCloud<pcl::PointXYZ>& input, double voxel_size) {
  index_vector.clear();
  index_vector.reserve(input.points.size());

  // First pass: go over all points and insert them into the index_vector vector
  // with calculated idx. Points with the same idx value will contribute to the
  // same point of resulting CloudPoint
  for (int i = 0; i < input.points.size(); i++) {
    unsigned int x = round(input.points[i].x / voxel_size) + 512 - 1;  // offset 512 - 1
    unsigned int y = round(input.points[i].y / voxel_size) + 512 - 1;
    unsigned int z = round(input.points[i].z / voxel_size) + 512 - 1;

    // hashing
    unsigned int rx = (x << 20) & 0x3FF00000;
    unsigned int ry = (y << 10) & 0x000FFC00;
    unsigned int rz = z & 0x000003FF;
    unsigned int hash = rx + ry + rz;

    index_vector.emplace_back(hash, i);
  }

  // Second pass: sort the index_vector vector using value representing target cell as index
  // in effect all points belonging to the same output cell will be next to each other
  std::sort(index_vector.begin(), index_vector.end(), std::less<cloud_point_index_idx>());
}

void FeatureExtractor::Voxelize(pcl::PointCloud<pcl::PointXYZ>& input, pcl::PointCloud<pcl::PointXYZ>& output,
                                double voxel_size, unsigned int min_points_per_voxel) {
  // Third pass: count output cells
  // we need to skip all the same, adjacent idx values
  unsigned int total = 0;
  unsigned int index = 0;
  std::vector<int> first_indices_vector;
  v_index_vector.clear();
  v_index_vector.reserve(output.points.size());
  while (index < index_vector.size()) {
    unsigned int i = index + 1;
    while (i < index_vector.size() && index_vector[i].idx == index_vector[index].idx) ++i;
    if (i - index >= min_points_per_voxel) {
      ++total;
      first_indices_vector.emplace_back(index);
      // <hash, index of voxel vector>
      v_index_vector.emplace_back(index_vector[index].idx, first_indices_vector.size() - 1);
    }
    index = i;
  }

  // Fourth pass: insert voxels into the output
  output.points.reserve(total);
  for (int first_idx : first_indices_vector) {
    // unhashing
    double x = (int((index_vector[first_idx].idx & 0x3FF00000) >> 20) - (512 - 1)) * voxel_size;
    double y = (int((index_vector[first_idx].idx & 0x000FFC00) >> 10) - (512 - 1)) * voxel_size;
    double z = (int(index_vector[first_idx].idx & 0x000003FF) - (512 - 1)) * voxel_size;

    output.points.emplace_back(x, y, z);
  }
  output.width = static_cast<std::uint32_t>(output.points.size());
  output.height = 1;       // downsampling breaks the organized structure
  output.is_dense = true;  // we filter out invalid points
}

void FeatureExtractor::ExtractLine(pcl::PointCloud<pcl::PointXYZ>& v_input,
                                   std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>>& output) {
  int idx1 = 0, idx2 = 1;
  while (idx2 < v_index_vector.size()) {
    if (v_index_vector[idx2].idx - v_index_vector[idx1].idx == idx2 - idx1) {
      idx2++;
    } else {
      if (idx2 - idx1 > 2) {
        pcl::PointXYZ p1, p2;
        p1.x = v_input.points[v_index_vector[idx1].voxel_index].x;
        p1.y = v_input.points[v_index_vector[idx1].voxel_index].y;
        p1.z = v_input.points[v_index_vector[idx1].voxel_index].z;
        p2.x = v_input.points[v_index_vector[idx2 - 1].voxel_index].x;
        p2.y = v_input.points[v_index_vector[idx2 - 1].voxel_index].y;
        p2.z = v_input.points[v_index_vector[idx2 - 1].voxel_index].z;
        output.emplace_back(p1, p2);
      }
      idx1 = idx2;
      idx2++;
    }
  }
}

void FeatureExtractor::HSVtoRGB(int h, int s, int v, int& r, int& g, int& b) {
  int i2 = h / 43;
  int remainder = (h - (i2 * 43)) * 6;

  int p = (v * (255 - s)) >> 8;
  int q = (v * (255 - ((s * remainder) >> 8))) >> 8;
  int t = (v * (255 - ((s * (255 - remainder)) >> 8))) >> 8;

  switch (i2) {
    case 0:
      r = v;
      g = t;
      b = p;
      break;
    case 1:
      r = q;
      g = v;
      b = p;
      break;
    case 2:
      r = p;
      g = v;
      b = t;
      break;
    case 3:
      r = p;
      g = q;
      b = v;
      break;
    case 4:
      r = t;
      g = p;
      b = v;
      break;
    case 5:
      r = v;
      g = p;
      b = q;
      break;
  }
}

void FeatureExtractor::VisualizeVoxel(pcl::PointCloud<pcl::PointXYZ>& input, double voxel_size) {
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker marker;
  for (int i = 0; i < input.points.size(); i++) {
    double x = input.points[i].x;
    double y = input.points[i].y;
    double z = input.points[i].z;

    marker = visualization_msgs::Marker();
    // Set the frame ID and timestamp.
    marker.header.frame_id = "velo_link";
    marker.header.stamp = ros::Time::now();
    // Set the namespace and id for this marker. This serves to create a unique ID Any marker sent with the same
    // namespace and id will overwrite the old one
    marker.ns = "point";
    marker.id = i;
    // Set the marker type. Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    marker.type = visualization_msgs::Marker::CUBE;
    // Set the marker action. Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker.action = visualization_msgs::Marker::ADD;
    // Set the pose of the marker. This is a full 6DOF pose relative to the frame/time specified in the header
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = z;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    // Set the scale of the marker -- 1x1x1 here means 1m on a side
    marker.scale.x = voxel_size;
    marker.scale.y = voxel_size;
    marker.scale.z = voxel_size;

    int h = i / float(input.points.size() - 1) * 255;
    int s = 255;
    int v = 255;
    int r = 0, g = 0, b = 0;
    HSVtoRGB(h, s, v, r, g, b);

    marker.color.r = r / 255.0;
    marker.color.g = g / 255.0;
    marker.color.b = b / 255.0;
    marker.color.a = 0.5;

    marker.lifetime = ros::Duration();

    marker_array.markers.push_back(marker);
  }

  point_pub_.publish(marker_array);
}

void FeatureExtractor::VisualizeLine(std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>>& lines) {
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker marker;
  for (int i = 0; i < lines.size(); i++) {
    double x1 = lines[i].first.x;
    double y1 = lines[i].first.y;
    double z1 = lines[i].first.z;
    double x2 = lines[i].second.x;
    double y2 = lines[i].second.y;
    double z2 = lines[i].second.z;

    marker = visualization_msgs::Marker();
    // Set the frame ID and timestamp.
    marker.header.frame_id = "velo_link";
    marker.header.stamp = ros::Time::now();
    // Set the namespace and id for this marker. This serves to create a unique ID Any marker sent with the same
    // namespace and id will overwrite the old one
    marker.ns = "line";
    marker.id = i;
    // Set the marker type. Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    // Set the marker action. Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker.action = visualization_msgs::Marker::ADD;
    // Set the pose of the marker. only w component is used for LINE_STRIP markers
    marker.pose.orientation.w = 1.0;
    // Set the scale of the marker. only x component is used for LINE_STRIP markers
    marker.scale.x = 0.1;

    int h = 0;
    int s = 255;
    int v = 255;
    int r = 0, g = 0, b = 0;
    HSVtoRGB(h, s, v, r, g, b);

    marker.color.r = r / 255.0;
    marker.color.g = g / 255.0;
    marker.color.b = b / 255.0;
    marker.color.a = 0.5;

    marker.lifetime = ros::Duration();

    geometry_msgs::Point p1, p2;
    p1.x = x1;
    p1.y = y1;
    p1.z = z1;
    p2.x = x2;
    p2.y = y2;
    p2.z = z2;
    marker.points.push_back(p1);
    marker.points.push_back(p2);

    marker_array.markers.push_back(marker);
  }
  line_pub_.publish(marker_array);
}

void FeatureExtractor::VisualizeLineDensity(std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>>& lines,
                                            double voxel_size) {
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker marker;
  for (int i = 0; i < lines.size(); i++) {
    double x = lines[i].first.x;
    double y = lines[i].first.y;
    double z1 = lines[i].first.z;
    double z2 = lines[i].second.z;

    // line in the same voxel
    if (lines[i + 1].first.x == x && lines[i + 1].first.y == y) {
      z2 += lines[i + 1].second.z - lines[i + 1].first.z;
      i++;
    }

    marker = visualization_msgs::Marker();
    // Set the frame ID and timestamp.
    marker.header.frame_id = "velo_link";
    marker.header.stamp = ros::Time::now();
    // Set the namespace and id for this marker. This serves to create a unique ID Any marker sent with the same
    // namespace and id will overwrite the old one
    marker.ns = "line_density";
    marker.id = i;
    // Set the marker type. Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    marker.type = visualization_msgs::Marker::CUBE;
    // Set the marker action. Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = -1.73;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = voxel_size;
    marker.scale.y = voxel_size;
    marker.scale.z = 0.001;

    int h = (z2 - z1) * 255 / voxel_size / 20;
    int s = 255;
    int v = 255;
    int r = 0, g = 0, b = 0;
    HSVtoRGB(h, s, v, r, g, b);

    marker.color.r = r / 255.0;
    marker.color.g = g / 255.0;
    marker.color.b = b / 255.0;
    marker.color.a = 0.5;

    marker.lifetime = ros::Duration();

    marker_array.markers.push_back(marker);
  }
  line_density_pub_.publish(marker_array);
}

HeightGrid FeatureExtractor::GetHeightGridFromLines(std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>>& lines,
                                                    double voxel_size) {
  HeightGrid hg(ros::Time::now().toSec(), voxel_size, 1024, 1024);
  std::vector<Cell> cells;

  for (int i = 0; i < lines.size(); i++) {
    double x = lines[i].first.x;
    double y = lines[i].first.y;
    double z1 = lines[i].first.z;
    double z2 = lines[i].second.z;

    // line in the same voxel
    if (lines[i + 1].first.x == x && lines[i + 1].first.y == y) {
      z2 += lines[i + 1].second.z - lines[i + 1].first.z;
      i++;
    }

    cells.emplace_back(x, y, z2 - z1);
  }

  hg.SetCells(cells);

  return hg;
}

void FeatureExtractor::VisualizeHeightGrid(HeightGrid& height_grid) {
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker marker;

  for (int i = 0; i < height_grid.GetCells().size(); i++) {
    double x = height_grid.GetCells()[i].x;
    double y = height_grid.GetCells()[i].y;
    double height = height_grid.GetCells()[i].height;

    marker = visualization_msgs::Marker();
    // Set the frame ID and timestamp.
    marker.header.frame_id = "velo_link";
    marker.header.stamp = ros::Time(height_grid.GetTimestamp());
    // Set the namespace and id for this marker. This serves to create a unique ID Any marker sent with the same
    // namespace and id will overwrite the old one
    marker.ns = "height_grid";
    marker.id = i;
    // Set the marker type. Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    marker.type = visualization_msgs::Marker::CUBE;
    // Set the marker action. Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker.action = visualization_msgs::Marker::ADD;
    // Set the pose of the marker. This is a full 6DOF pose relative to the frame/time specified in the header
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = -1.73;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    // Set the scale of the marker -- 1x1x1 here means 1m on a side
    marker.scale.x = height_grid.GetResolution();
    marker.scale.y = height_grid.GetResolution();
    marker.scale.z = 0.001;

    int h = height * 255 / height_grid.GetResolution() / 20;
    int s = 255;
    int v = 255;
    int r = 0, g = 0, b = 0;
    HSVtoRGB(h, s, v, r, g, b);

    marker.color.r = r / 255.0;
    marker.color.g = g / 255.0;
    marker.color.b = b / 255.0;
    marker.color.a = 0.5;

    marker.lifetime = ros::Duration();

    marker_array.markers.push_back(marker);
  }

  height_grid_pub_.publish(marker_array);
}

void FeatureExtractor::VisualizeHeightGridInOccupancyGrid(HeightGrid& height_grid) {
  nav_msgs::OccupancyGrid grid = height_grid.ToOccupancyGrid();
  height_grid_occ_pub_.publish(grid);
}