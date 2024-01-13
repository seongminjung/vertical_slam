#include "vertical_slam/feature_extractor.h"

#include <geometry_msgs/Point.h>
#include <nav_msgs/OccupancyGrid.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
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

    // line in the same cell
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

    // line in the same cell
    if (lines[i + 1].first.x == x && lines[i + 1].first.y == y) {
      z2 += lines[i + 1].second.z - lines[i + 1].first.z;
      i++;
    }

    cells.emplace_back(x, y, z2 - z1);
  }

  hg.SetCells(cells);

  return hg;
}

void FeatureExtractor::VisualizeHeightGrid(HeightGrid& height_grid, int color) {
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker marker;

  for (int i = 0; i < height_grid.GetCells().size(); i++) {
    double x = height_grid.GetCells()[i].x;
    double y = height_grid.GetCells()[i].y;
    double height = height_grid.GetCells()[i].height;

    bool disabled = height_grid.GetCells()[i].disabled;

    marker = visualization_msgs::Marker();
    // Set the frame ID and timestamp.
    marker.header.frame_id = "velo_link";
    marker.header.stamp = ros::Time(height_grid.GetTimestamp());
    // Set the namespace and id for this marker. This serves to create a unique ID Any marker sent with the same
    // namespace and id will overwrite the old one
    if (color == 0) {
      marker.ns = "height_grid_0";
    } else if (color == 1) {
      marker.ns = "height_grid_1";
    } else if (color == 2) {
      marker.ns = "height_grid_2";
    }
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

    if (color == 0) {
      marker.color.r = 1;
      marker.color.g = 0;
      marker.color.b = 0;
    } else if (color == 1) {
      marker.color.r = 0;
      marker.color.g = 1;
      marker.color.b = 0;
    } else if (color == 2) {
      marker.color.r = 0;
      marker.color.g = 0;
      marker.color.b = 1;
    }

    marker.color.a = 0.5 * height;

    // if disabled, make it yellow
    if (disabled) {
      marker.color.r = 1;
      marker.color.g = 1;
      marker.color.b = 0;
      marker.color.a = 0.5;
    }

    marker.lifetime = ros::Duration();

    marker_array.markers.push_back(marker);
  }

  height_grid_pub_.publish(marker_array);
}

void FeatureExtractor::VisualizeLineBetweenMatchingPoints(HeightGrid HG1, HeightGrid HG2) {
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker marker;

  std::vector<Cell> cells1 = HG1.GetCells();
  std::vector<Cell> cells2 = HG2.GetCells();

  for (int i = 0; i < cells1.size(); i++) {
    double x1 = cells1[i].x;
    double y1 = cells1[i].y;
    double height1 = cells1[i].height;

    double x2 = cells2[i].x;
    double y2 = cells2[i].y;
    double height2 = cells2[i].height;

    bool disabled = cells1[i].disabled;

    marker = visualization_msgs::Marker();
    // Set the frame ID and timestamp.
    marker.header.frame_id = "velo_link";
    marker.header.stamp = ros::Time(HG1.GetTimestamp());
    // Set the namespace and id for this marker. This serves to create a unique ID Any marker sent with the same
    // namespace and id will overwrite the old one
    marker.ns = "line_between_matching_points";
    marker.id = i;
    // Set the marker type. Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    // Set the marker action. Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker.action = visualization_msgs::Marker::ADD;
    // Set the pose of the marker. only w component is used for LINE_STRIP markers
    marker.pose.orientation.w = 1.0;
    // Set the scale of the marker. only x component is used for LINE_STRIP markers
    marker.scale.x = 0.05;

    int h = 0;
    int s = 255;
    int v = 255;
    int r = 0, g = 0, b = 0;
    HSVtoRGB(h, s, v, r, g, b);

    marker.color.r = r / 255.0;
    marker.color.g = g / 255.0;
    marker.color.b = b / 255.0;
    marker.color.a = 0.5 * (height1 + height2);

    // if disabled, make it yellow
    if (disabled) {
      marker.color.r = 1;
      marker.color.g = 1;
      marker.color.b = 0;
      marker.color.a = 0.5;
    }

    marker.lifetime = ros::Duration();

    geometry_msgs::Point p1, p2;
    p1.x = x1;
    p1.y = y1;
    p1.z = -1.73;
    p2.x = x2;
    p2.y = y2;
    p2.z = -1.73;
    marker.points.push_back(p1);
    marker.points.push_back(p2);

    marker_array.markers.push_back(marker);

    height_grid_pub_.publish(marker_array);
  }
}

void FeatureExtractor::VisualizeCentroid(Eigen::Vector2d centroid, time_t timestamp, int color) {
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker marker;

  marker = visualization_msgs::Marker();
  // Set the frame ID and timestamp.
  marker.header.frame_id = "velo_link";
  marker.header.stamp = ros::Time(timestamp);
  // Set the namespace and id for this marker. This serves to create a unique ID Any marker sent with the same
  // namespace and id will overwrite the old one
  if (color == 0) {
    marker.ns = "centroid_0";
  } else if (color == 1) {
    marker.ns = "centroid_1";
  } else if (color == 2) {
    marker.ns = "centroid_2";
  }
  marker.id = color;
  // Set the marker type. Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
  marker.type = visualization_msgs::Marker::SPHERE;
  // Set the marker action. Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
  marker.action = visualization_msgs::Marker::ADD;
  // Set the pose of the marker. This is a full 6DOF pose relative to the frame/time specified in the header
  marker.pose.position.x = centroid(0);
  marker.pose.position.y = centroid(1);
  marker.pose.position.z = -1.73;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;

  // Set the scale of the marker -- 1x1x1 here means 1m on a side
  marker.scale.x = 0.1;
  marker.scale.y = 0.1;
  marker.scale.z = 0.1;

  if (color == 0) {
    marker.color.r = 1;
    marker.color.g = 0;
    marker.color.b = 0;
  } else if (color == 1) {
    marker.color.r = 0;
    marker.color.g = 1;
    marker.color.b = 0;
  } else if (color == 2) {
    marker.color.r = 0;
    marker.color.g = 0;
    marker.color.b = 1;
  }

  marker.color.a = 0.5;

  marker.lifetime = ros::Duration();

  marker_array.markers.push_back(marker);

  height_grid_pub_.publish(marker_array);
}

void FeatureExtractor::VisualizeHeightGridInOccupancyGrid(HeightGrid& height_grid) {
  nav_msgs::OccupancyGrid grid = height_grid.ToOccupancyGrid();
  height_grid_occ_pub_.publish(grid);
}

Point FeatureExtractor::GetCentroid(HeightGrid& height_grid) {
  std::vector<Cell> cells = height_grid.GetCells();
  double x_sum = 0;
  double y_sum = 0;
  double height_sum = 0;
  for (int i = 0; i < cells.size(); i++) {
    x_sum += cells[i].x * cells[i].height;
    y_sum += cells[i].y * cells[i].height;
    height_sum += cells[i].height;
  }
  Point centroid(x_sum / height_sum, y_sum / height_sum);
  return centroid;
}

Point FeatureExtractor::GetCentroidWithoutHeight(HeightGrid& height_grid) {
  std::vector<Cell> cells = height_grid.GetCells();
  double x_sum = 0;
  double y_sum = 0;
  double height_sum = 0;
  for (int i = 0; i < cells.size(); i++) {
    x_sum += cells[i].x;
    y_sum += cells[i].y;
    height_sum += cells[i].height;
  }
  Point centroid(x_sum / cells.size(), y_sum / cells.size());
  return centroid;
}

void FeatureExtractor::DemeanHeightGrid(HeightGrid& height_grid_in, HeightGrid& height_grid_out, Point centroid) {
  std::vector<Cell> cells_in = height_grid_in.GetCells();
  std::vector<Cell> cells_out;
  cells_out.reserve(cells_in.size());
  for (int i = 0; i < cells_in.size(); i++) {
    cells_out.emplace_back(cells_in[i].x - centroid.x, cells_in[i].y - centroid.y, cells_in[i].height);
  }
  height_grid_out.SetTimestamp(height_grid_in.GetTimestamp());
  height_grid_out.SetResolution(height_grid_in.GetResolution());
  height_grid_out.SetWidth(height_grid_in.GetWidth());
  height_grid_out.SetHeight(height_grid_in.GetHeight());
  height_grid_out.SetCells(cells_out);
}

void FeatureExtractor::RunICP(HeightGrid& M, HeightGrid& P) {
  /// \brief 2D HeightGrid ICP algorithm
  /// \param M: map HeightGrid
  /// \param P: incoming HeightGrid to be aligned

  // Initialization
  Eigen::Matrix2d accumulated_R = Eigen::Matrix2d::Identity();  // rotation
  Eigen::Vector2d accumulated_t = Eigen::Vector2d::Zero();      // translation
  Eigen::Matrix2d R = Eigen::Matrix2d::Identity();  // rotation
  Eigen::Vector2d t = Eigen::Vector2d::Zero();      // translation
  double err = 0;                                   // error
  HeightGrid new_P(P);                              // transformed P
  int max_iter = 200;
  double thresh = 1e-5;
  std::vector<std::pair<int, double>> dist_vector;  // <index of new_P, distance>

  int Nm = M.GetCells().size();
  int Np = new_P.GetCells().size();

  // Start ICP Loop
  for (int iter = 0; iter < max_iter; iter++) {
    ROS_INFO("==========iter: %d==========", iter);
    HeightGrid Y;
    Y.ReserveCells(Np);
    dist_vector.clear();
    dist_vector.reserve(Np);

    // Revoke disabled status
    for (int i = 0; i < Np; i++) {
      new_P.UpdateOneCellDisabled(i, false);
    }

    // Find the nearest neighbor for each point in P
    for (int i = 0; i < Np; i++) {
      double min_dist = 1e10;
      int min_idx = 0;
      for (int j = 0; j < Nm; j++) {
        double dist = sqrt(pow(new_P.GetCells()[i].x - M.GetCells()[j].x, 2) +
                           pow(new_P.GetCells()[i].y - M.GetCells()[j].y, 2));  // Euclidean distance
        if (dist < min_dist) {
          // Update only when height is similar
          if (abs(new_P.GetCells()[i].height - M.GetCells()[j].height) < 0.5) {
            min_dist = dist;
            min_idx = j;
          }
        }
      }
      dist_vector.emplace_back(i, min_dist);
      Y.AppendOneCell(M.GetCells()[min_idx]);
    }

    // print histogram. x: distance, y: count
    std::vector<int> hist(100, 0);

    // fill with 0
    for (int i = 0; i < hist.size(); i++) {
      hist[i] = 0;
    }

    for (int i = 0; i < Np; i++) {
      hist[std::min(99, int(dist_vector[i].second * 50))]++;
    }

    for (int i = 0; i < hist.size(); i++) {
      std::cout << hist[i] << " ";
    }
    std::cout << std::endl;

    // sort dist_vector by distance
    std::sort(dist_vector.begin(), dist_vector.end(),
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) { return a.second > b.second; });

    // Drop points with top 10% distance by making height zero
    for (int i = 0; i < Np * 0.1; i++) {
      new_P.UpdateOneCellDisabled(dist_vector[i].first, true);
    }

    // // if less than 80% of the pairs have distance < 0.01, drop the point with distance < 0.01 by making height zero
    // // since it means overlapping
    // int count = 0;
    // for (int i = 0; i < Np; i++) {
    //   if (dist_vector[i].second < 0.01) {
    //     count++;
    //   }
    // }
    // if (count < Np * 0.8) {
    //   for (int i = 0; i < Np; i++) {
    //     if (dist_vector[i].second < 0.01) {
    //       new_P.UpdateOneCellDisabled(dist_vector[i].first, true);
    //     }
    //   }
    // }

    // Visualize
    VisualizeLineBetweenMatchingPoints(new_P, Y);
    // ros::Duration(1).sleep(); //Only for RunTestICP
    VisualizeHeightGrid(new_P, 2);

    Eigen::Matrix3d result = FindAlignment(new_P, Y);  // left top 2x2: R, right top 2x1: t, left bottom 1x1: err

    std::cout << "R_raw: " << std::endl << result.block<2, 2>(0, 0) << std::endl;
    std::cout << "t_raw: " << std::endl << result.block<2, 1>(0, 2) << std::endl;

    // Update R, t, err
    R = result.block<2, 2>(0, 0);
    t = result.block<2, 1>(0, 2);
    accumulated_R = R * accumulated_R;
    accumulated_t = R * accumulated_t + t;
    err = result(2, 0);

    // Print R, t, err
    std::cout << "R: " << std::endl << accumulated_R << std::endl;
    std::cout << "t: " << std::endl << accumulated_t << std::endl;
    std::cout << "err: " << std::endl << err << std::endl;

    // Update P and compute error
    for (int i = 0; i < Np; i++) {
      Cell new_cell(R(0, 0) * new_P.GetCells()[i].x + R(0, 1) * new_P.GetCells()[i].y + t(0),
                    R(1, 0) * new_P.GetCells()[i].x + R(1, 1) * new_P.GetCells()[i].y + t(1),
                    new_P.GetCells()[i].height);
      new_P.UpdateOneCell(i, new_cell);
    }

    // Check for convergence
    if (err < thresh) {
      VisualizeLineBetweenMatchingPoints(new_P, Y);
      break;
    }
  }
}

void FeatureExtractor::RunTestICP() {
  HeightGrid X;  // incoming
  X.SetTimestamp(ros::Time::now().toSec());
  X.SetResolution(0.2);
  X.SetWidth(2);
  X.SetHeight(2);
  X.ReserveCells(4);
  X.AppendOneCell(Cell(3, -5, 3));
  X.AppendOneCell(Cell(2, -5, 3));
  X.AppendOneCell(Cell(1, -4, 1));
  X.AppendOneCell(Cell(1, -3, 1));
  X.AppendOneCell(Cell(2, -2, 2));
  X.AppendOneCell(Cell(3, -2, 2));
  X.AppendOneCell(Cell(4, -3, 1));
  X.AppendOneCell(Cell(4, -4, 1));
  X.AppendOneCell(Cell(3, 0, 3));
  X.AppendOneCell(Cell(2, 0, 3));
  X.AppendOneCell(Cell(1, 1, 1));
  X.AppendOneCell(Cell(1, 2, 1));
  X.AppendOneCell(Cell(2, 3, 2));
  X.AppendOneCell(Cell(3, 3, 2));
  X.AppendOneCell(Cell(4, 2, 1));
  X.AppendOneCell(Cell(4, 1, 1));
  VisualizeHeightGrid(X, 1);

  HeightGrid Y;  // map
  Y.SetTimestamp(ros::Time::now().toSec());
  Y.SetResolution(0.2);
  Y.SetWidth(2);
  Y.SetHeight(2);
  Y.ReserveCells(4);
  Y.AppendOneCell(Cell(0, 0, 1));
  Y.AppendOneCell(Cell(0, 1, 1));
  Y.AppendOneCell(Cell(1, 2, 1));
  Y.AppendOneCell(Cell(2, 2, 1));
  Y.AppendOneCell(Cell(3, 1, 3));
  Y.AppendOneCell(Cell(3, 0, 3));
  Y.AppendOneCell(Cell(2, -1, 1));
  Y.AppendOneCell(Cell(1, -1, 1));
  Y.AppendOneCell(Cell(5, 0, 1));
  Y.AppendOneCell(Cell(5, 1, 1));
  Y.AppendOneCell(Cell(6, 2, 1));
  Y.AppendOneCell(Cell(7, 2, 1));
  Y.AppendOneCell(Cell(8, 1, 3));
  Y.AppendOneCell(Cell(8, 0, 3));
  Y.AppendOneCell(Cell(7, -1, 1));
  Y.AppendOneCell(Cell(6, -1, 1));
  VisualizeHeightGrid(Y, 0);

  RunICP(Y, X);
}

Eigen::Matrix3d FeatureExtractor::FindAlignment(HeightGrid& X_HG, HeightGrid& Y_HG) {
  /// \brief Find the alignment between X and Y
  /// \param X_HG: transformed X from last iteration
  /// \param Y_HG: nearest neighbor of each point in X
  /// \return result: left top 2x2: R, right top 2x1: t, left bottom 1x1 s, center bottom 1x1: err

  // Test the inputs
  if (X_HG.GetCells().size() != Y_HG.GetCells().size()) {
    ROS_ERROR("X and Y have different sizes!");
  }
  if (X_HG.GetCells().size() < 4) {
    ROS_ERROR("Need at least four pairs of points!");
  }

  unsigned int N_og = X_HG.GetCells().size();

  Eigen::MatrixXd X_matrix_og = X_HG.ToEigenMatrix();
  Eigen::MatrixXd Y_matrix_og = Y_HG.ToEigenMatrix();

  // Make new X and Y matrix without disabled cells
  Eigen::MatrixXd X_matrix;
  Eigen::MatrixXd Y_matrix;
  for (int i = 0; i < N_og; i++) {
    if (!X_HG.GetCells()[i].disabled) {
      X_matrix.conservativeResize(3, X_matrix.cols() + 1);
      X_matrix.col(X_matrix.cols() - 1) = X_matrix_og.col(i);
      Y_matrix.conservativeResize(3, Y_matrix.cols() + 1);
      Y_matrix.col(Y_matrix.cols() - 1) = Y_matrix_og.col(i);
    }
  }

  unsigned int N = X_matrix.cols();

  // Seperate coordinates and height
  Eigen::MatrixXd X = X_matrix.block(0, 0, 2, N);
  Eigen::MatrixXd Y = Y_matrix.block(0, 0, 2, N);
  Eigen::VectorXd X_height = X_matrix.block(2, 0, 1, N).transpose();
  Eigen::VectorXd Y_height = Y_matrix.block(2, 0, 1, N).transpose();

  // Compute the centroid of X and Y, weighted by height
  Eigen::Vector2d X_centroid = X * X_height / X_height.sum();
  Eigen::Vector2d Y_centroid = Y * Y_height / Y_height.sum();

  // Compute average height of X and Y
  Eigen::VectorXd Height = (X_height + Y_height) / 2;

  // Compute the demeaned X and Y
  Eigen::MatrixXd X_demeaned = X - X_centroid * Eigen::MatrixXd::Ones(1, N);
  Eigen::MatrixXd Y_demeaned = Y - Y_centroid * Eigen::MatrixXd::Ones(1, N);

  // Compute the covariance matrix including height
  Eigen::Matrix2d H = X_demeaned * Height.asDiagonal() * Y_demeaned.transpose();

  // Compute the SVD of H
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix2d U = svd.matrixU();
  Eigen::Matrix2d V = svd.matrixV();
  Eigen::Matrix2d R = V * U.transpose();

  // get angle in degrees from rotation matrix r
  double angle = atan2(R(1, 0), R(0, 0)) * 180 / M_PI;

  // Compute the translation
  Eigen::Vector2d t = Y_centroid - R * X_centroid;

  // Compute the error
  double err = (Y_demeaned - R * X_demeaned).norm() / N;

  // Construct the result
  Eigen::Matrix3d result;
  result.block<2, 2>(0, 0) = R;
  result.block<2, 1>(0, 2) = t;
  result(2, 0) = err;

  // Visualize centroid of X, Y, and converted X
  VisualizeCentroid(Y_centroid, X_HG.GetTimestamp(), 0);
  VisualizeCentroid(X_centroid, X_HG.GetTimestamp(), 1);
  Eigen::Vector2d X_centroid_tf = R * X_centroid + t;
  VisualizeCentroid(X_centroid_tf, X_HG.GetTimestamp(), 2);

  // // Print coordinates of each centroids
  // ROS_INFO("X_centroid: \n%f\n%f", X_centroid(0), X_centroid(1));
  // ROS_INFO("Y_centroid: \n%f\n%f", Y_centroid(0), Y_centroid(1));
  // ROS_INFO("X_centroid_tf: \n%f\n%f", X_centroid_tf(0), X_centroid_tf(1));

  return result;
}