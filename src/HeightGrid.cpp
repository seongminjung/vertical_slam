#include "vertical_slam/HeightGrid.h"

#include <nav_msgs/OccupancyGrid.h>

#include <algorithm>

HeightGrid::HeightGrid() {}

HeightGrid::HeightGrid(time_t timestamp, double resolution, unsigned int width, unsigned int height)
    : timestamp_(timestamp), resolution_(resolution), width_(width), height_(height) {}

HeightGrid::~HeightGrid() {}

/////////////////////////////////////////
////////// Setters and Getters //////////
void HeightGrid::SetTimestamp(time_t timestamp) { timestamp_ = timestamp; }

void HeightGrid::SetResolution(double resolution) { resolution_ = resolution; }

void HeightGrid::SetWidth(unsigned int width) { width_ = width; }

void HeightGrid::SetHeight(unsigned int height) { height_ = height; }  // height of the map itself

void HeightGrid::SetCells(std::vector<Cell> cells) { cells_ = cells; }

time_t HeightGrid::GetTimestamp() { return timestamp_; }

double HeightGrid::GetResolution() { return resolution_; }

unsigned int HeightGrid::GetWidth() { return width_; }

unsigned int HeightGrid::GetHeight() { return height_; }  // height of the map itself

std::vector<Cell> HeightGrid::GetCells() { return cells_; }
/////////////////////////////////////////
/////////////////////////////////////////

nav_msgs::OccupancyGrid HeightGrid::ToOccupancyGrid() {
  nav_msgs::OccupancyGrid grid;
  grid.header.stamp = ros::Time(timestamp_);
  grid.header.frame_id = "velo_link";
  grid.info.resolution = resolution_;
  grid.info.width = width_;
  grid.info.height = height_;
  grid.info.origin.position.x = 0.0 - (512 - 1) * resolution_;
  grid.info.origin.position.y = 0.0 - (512 - 1) * resolution_;
  grid.info.origin.position.z = -1.73;
  grid.info.origin.orientation.x = 0.0;
  grid.info.origin.orientation.y = 0.0;
  grid.info.origin.orientation.z = 0.0;
  grid.info.origin.orientation.w = 1.0;
  grid.data.resize(width_ * height_);

  // fill with zeros
  std::fill(grid.data.begin(), grid.data.end(), 0);

  for (int i = 0; i < cells_.size(); i++) {
    int x = cells_[i].x / resolution_ + (512 - 1);
    int y = cells_[i].y / resolution_ + (512 - 1);
    int index = y * width_ + x;
    grid.data[index] = cells_[i].height * 10;
  }

  return grid;
}