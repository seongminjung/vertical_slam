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
  double s = 1;                                     // scale
  Eigen::Matrix2d R = Eigen::Matrix2d::Identity();  // rotation
  Eigen::Vector2d t = Eigen::Vector2d::Zero();      // translation
  double err = 0;                                   // error
  HeightGrid new_P(P);                              // transformed P
  int max_iter = 200;
  double thresh = 1e-5;

  int Nm = M.GetCells().size();
  int Np = P.GetCells().size();
  int dim = 3;  // Although we are working in 2D, we use 3D vectors for now

  // Start ICP Loop
  for (int iter = 0; iter < max_iter; iter++) {
    HeightGrid Y;
    Y.ReserveCells(Np);

    // Find the nearest neighbor for each point in P
    for (int i = 0; i < Np; i++) {
      double min_dist = 1e10;
      int min_idx = 0;
      for (int j = 0; j < Nm; j++) {
        double dist = sqrt(pow(new_P.GetCells()[i].x - M.GetCells()[j].x, 2) +
                           pow(new_P.GetCells()[i].y - M.GetCells()[j].y, 2));  // Euclidean distance
        if (dist < min_dist) {
          min_dist = dist;
          min_idx = j;
        }
      }
      Y.AppendOneCell(M.GetCells()[min_idx]);
    }

    Eigen::Matrix3d result = FindAlignment(new_P, Y);  // left top 2x2: R, right top 2x1: t, left bottom 1x1: s,
                                                       // center bottom 1x1: err

    // Update R, t, s, err
    R = result.block<2, 2>(0, 0);
    t = result.block<2, 1>(0, 2);
    s = result(2, 0);
    err = result(2, 1);

    // Update P and compute error
    for (int i = 0; i < Np; i++) {
      Cell new_cell(s * R(0, 0) * new_P.GetCells()[i].x + s * R(0, 1) * new_P.GetCells()[i].y + t(0),
                    s * R(1, 0) * new_P.GetCells()[i].x + s * R(1, 1) * new_P.GetCells()[i].y + t(1),
                    new_P.GetCells()[i].height);
      new_P.UpdateOneCell(i, new_cell);
    }

    // Check for convergence
    if (err < thresh) {
      break;
    }

    // Visualize
    VisualizeHeightGrid(new_P, 2);

    ROS_INFO("R: \n%f, %f\n%f, %f", R(0, 0), R(0, 1), R(1, 0), R(1, 1));
    ROS_INFO("t: \n%f\n%f", t(0), t(1));
    ROS_INFO("s: %f", s);

    ROS_INFO("iter: %d, err: %f", iter, err);
  }
}

Eigen::Matrix3d FeatureExtractor::FindAlignment(HeightGrid& P, HeightGrid& Y) {
  /// \brief Find the alignment between P and Y
  /// \param P: transformed P from last iteration
  /// \param Y: nearest neighbor of each point in P
  /// \return result: left top 2x2: R, right top 2x1: t, left bottom 1x1 s, center bottom 1x1: err

  // Test the inputs
  if (P.GetCells().size() != Y.GetCells().size()) {
    ROS_ERROR("P and Y have different sizes!");
  }
  if (P.GetCells().size() < 4) {
    ROS_ERROR("Need at least four pairs of points!");
  }

  // Initialization
  int N = P.GetCells().size();

  // Compute the centroid of P and Y
  Point Mu_P_Point = GetCentroidWithoutHeight(P);
  Point Mu_Y_Point = GetCentroidWithoutHeight(Y);
  Eigen::Vector2d Mu_P(Mu_P_Point.x, Mu_P_Point.y);
  Eigen::Vector2d Mu_Y(Mu_Y_Point.x, Mu_Y_Point.y);

  // Compute the demeaned P and Y
  HeightGrid PprimeHG;
  HeightGrid YprimeHG;
  DemeanHeightGrid(P, PprimeHG, Mu_P_Point);
  DemeanHeightGrid(Y, YprimeHG, Mu_Y_Point);

  // Convert HeightGrid to Eigen::Matrix
  Eigen::MatrixXd Pprime;
  Eigen::MatrixXd Yprime;
  Pprime.resize(3, N);
  Yprime.resize(3, N);

  for (int i = 0; i < N; i++) {
    Pprime(0, i) = PprimeHG.GetCells()[i].x;
    Pprime(1, i) = PprimeHG.GetCells()[i].y;
    Yprime(0, i) = YprimeHG.GetCells()[i].x;
    Yprime(1, i) = YprimeHG.GetCells()[i].y;
  }

  // Compute the optimal quaternion
  // Eigen::MatrixXd Px: x component of Pprime
  // Eigen::MatrixXd Py: y component of Pprime
  // Eigen::MatrixXd Pz: zeros
  // Eigen::MatrixXd Yx: x component of Yprime
  // Eigen::MatrixXd Yy: y component of Yprime
  // Eigen::MatrixXd Yz: zeros
  Eigen::MatrixXd Px;
  Eigen::MatrixXd Py;
  Eigen::MatrixXd Pz;
  Eigen::MatrixXd Yx;
  Eigen::MatrixXd Yy;
  Eigen::MatrixXd Yz;
  Px.resize(3, N);
  Py.resize(3, N);
  Pz.resize(3, N);
  Yx.resize(3, N);
  Yy.resize(3, N);
  Yz.resize(3, N);

  for (int i = 0; i < N; i++) {
    Px(0, i) = Pprime(0, i);
    Px(1, i) = 0;
    Px(2, i) = 0;
    Py(0, i) = Pprime(1, i);
    Py(1, i) = 0;
    Py(2, i) = 0;
    Pz(0, i) = Pprime(2, i);
    Pz(1, i) = 0;
    Pz(2, i) = 0;
    Yx(0, i) = Yprime(0, i);
    Yx(1, i) = 0;
    Yx(2, i) = 0;
    Yy(0, i) = Yprime(1, i);
    Yy(1, i) = 0;
    Yy(2, i) = 0;
    Yz(0, i) = Yprime(2, i);
    Yz(1, i) = 0;
    Yz(2, i) = 0;
  }

  // Sum components
  Eigen::MatrixXd Sxx = Px * Yx.transpose();
  Eigen::MatrixXd Sxy = Px * Yy.transpose();
  Eigen::MatrixXd Sxz = Px * Yz.transpose();
  Eigen::MatrixXd Syx = Py * Yx.transpose();
  Eigen::MatrixXd Syy = Py * Yy.transpose();
  Eigen::MatrixXd Syz = Py * Yz.transpose();
  Eigen::MatrixXd Szx = Pz * Yx.transpose();
  Eigen::MatrixXd Szy = Pz * Yy.transpose();
  Eigen::MatrixXd Szz = Pz * Yz.transpose();

  // Construct the matrix Nmatrix
  Eigen::MatrixXd Nmatrix = Eigen::MatrixXd::Zero(4, 4);
  Nmatrix(0, 0) = Sxx.trace() + Syy.trace() + Szz.trace();
  Nmatrix(0, 1) = Syz.trace() - Szy.trace();
  Nmatrix(0, 2) = Szx.trace() - Sxz.trace();
  Nmatrix(0, 3) = Sxy.trace() - Syx.trace();
  Nmatrix(1, 0) = Nmatrix(0, 1);
  Nmatrix(1, 1) = Sxx.trace() - Syy.trace() - Szz.trace();
  Nmatrix(1, 2) = Sxy.trace() + Syx.trace();
  Nmatrix(1, 3) = Szx.trace() + Sxz.trace();
  Nmatrix(2, 0) = Nmatrix(0, 2);
  Nmatrix(2, 1) = Nmatrix(1, 2);
  Nmatrix(2, 2) = -Sxx.trace() + Syy.trace() - Szz.trace();
  Nmatrix(2, 3) = Syz.trace() + Szy.trace();
  Nmatrix(3, 0) = Nmatrix(0, 3);
  Nmatrix(3, 1) = Nmatrix(1, 3);
  Nmatrix(3, 2) = Nmatrix(2, 3);
  Nmatrix(3, 3) = -Sxx.trace() - Syy.trace() + Szz.trace();

  // Find the eigenvector corresponding to the largest eigenvalue
  Eigen::EigenSolver<Eigen::MatrixXd> es(Nmatrix);
  Eigen::MatrixXcd eigenvectors = es.eigenvectors();
  Eigen::VectorXcd eigenvalues = es.eigenvalues();

  // Find the index of the largest eigenvalue
  int max_idx = 0;
  double max_val = 0;
  for (int i = 0; i < 4; i++) {
    if (eigenvalues(i).real() > max_val) {
      max_val = eigenvalues(i).real();
      max_idx = i;
    }
  }

  // Find the optimal quaternion
  Eigen::Vector4d q = eigenvectors.col(max_idx).real();

  // Compute the rotation matrix
  Eigen::Matrix2d R;
  R(0, 0) = pow(q(0), 2) + pow(q(1), 2) - pow(q(2), 2) - pow(q(3), 2);
  R(0, 1) = 2 * (q(1) * q(2) - q(0) * q(3));
  R(1, 0) = 2 * (q(1) * q(2) + q(0) * q(3));
  R(1, 1) = pow(q(0), 2) - pow(q(1), 2) + pow(q(2), 2) - pow(q(3), 2);

  // Compute the scale
  double Sp = 0;
  double D = 0;
  for (int i = 0; i < N; i++) {
    Sp += PprimeHG.GetCells()[i].x * PprimeHG.GetCells()[i].x + PprimeHG.GetCells()[i].y * PprimeHG.GetCells()[i].y;
    D += YprimeHG.GetCells()[i].x * YprimeHG.GetCells()[i].x + YprimeHG.GetCells()[i].y * YprimeHG.GetCells()[i].y;
  }
  double s = sqrt(D / Sp);

  // Compute the translation vector
  Eigen::Vector2d t = Mu_Y - s * R * Mu_P;

  // Compute the error
  double err = 0;
  for (int i = 0; i < N; i++) {
    double e = pow(P.GetCells()[i].x - Y.GetCells()[i].x, 2) +
               pow(P.GetCells()[i].y - Y.GetCells()[i].y, 2);  // Sum of squared errors
    err += e;
  }

  // Construct the result matrix
  Eigen::Matrix3d result = Eigen::Matrix3d::Zero();
  result.block<2, 2>(0, 0) = R;
  result.block<2, 1>(0, 2) = t;
  result(2, 0) = s;
  result(2, 1) = err;

  return result;
}