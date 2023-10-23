#include <ros/ros.h>

#include "vertical_slam/feature_extractor.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "vertical_slam_node");
  FeatureExtractor feature_extractor;
  ros::spin();
  return 0;
}