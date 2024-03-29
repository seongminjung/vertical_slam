#ifndef HEIGHT_GRID_H
#define HEIGHT_GRID_H

#include <nav_msgs/OccupancyGrid.h>

#include <Eigen/Core>
#include <ctime>
#include <vector>

struct Point {
  double x;
  double y;
  Point() {}
  Point(double x_, double y_) : x(x_), y(y_) {}
};

struct Cell {
  double x;
  double y;
  double height;
  bool disabled;
  Cell() {}
  Cell(double x_, double y_, double height_) : x(x_), y(y_), height(height_), disabled(false) {}
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
  HeightGrid(const HeightGrid& other);  // copy constructor
  ~HeightGrid();

  void SetTimestamp(time_t timestamp);
  void SetResolution(double resolution);
  void SetWidth(unsigned int width);
  void SetHeight(unsigned int height);
  void SetCells(std::vector<Cell> cells);
  void AppendOneCell(Cell cell);

  void ReserveCells(unsigned int size);

  time_t GetTimestamp();
  double GetResolution();
  unsigned int GetWidth();
  unsigned int GetHeight();
  std::vector<Cell> GetCells();

  void UpdateOneCell(unsigned int index, Cell cell);
  void UpdateOneCellDisabled(unsigned int index, bool disabled);
  void RemoveOneCell(unsigned int index);

  Eigen::MatrixXd ToEigenMatrix();
  nav_msgs::OccupancyGrid ToOccupancyGrid();
};

#endif