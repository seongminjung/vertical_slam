#ifndef HEIGHT_GRID_H
#define HEIGHT_GRID_H

#include <ctime>
#include <vector>

struct Cell {
  double x;
  double y;
  double height;
  Cell() {}
  Cell(double x_, double y_, double height_) : x(x_), y(y_), height(height_) {}
};

class HeightGrid {
 private:
  time_t timestamp_;
  double resolution_;  // meter per cell
  unsigned int width_;
  unsigned int height_;
  std::vector<Cell> cells_;

 public:
  HeightGrid();
  HeightGrid(time_t timestamp, double resolution, unsigned int width, unsigned int height);
  ~HeightGrid();

  void SetTimestamp(time_t timestamp);
  void SetResolution(double resolution);
  void SetWidth(unsigned int width);
  void SetHeight(unsigned int height);
  void SetCells(std::vector<Cell> cells);

  time_t GetTimestamp();
  double GetResolution();
  unsigned int GetWidth();
  unsigned int GetHeight();
  std::vector<Cell> GetCells();
};

#endif