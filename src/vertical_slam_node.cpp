#include <ros/ros.h>

#include "vertical_slam/visualizer.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "vertical_slam_node");
  Visualizer visualizer;
  ros::spin();
  return 0;
}