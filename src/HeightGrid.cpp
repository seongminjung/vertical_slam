#include "vertical_slam/HeightGrid.h"

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