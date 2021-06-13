cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


add_executable( DisplayImage main.cpp DisplayImage.hpp bodyDetector.hpp backgroundRemover.hpp generic.hpp)
target_link_libraries( DisplayImage ${OpenCV_LIBS} )


add_executable( kmeansSample DisplayImage.hpp bodyDetector.hpp kmeans_sample.cpp backgroundRemover.hpp )
target_link_libraries( kmeansSample ${OpenCV_LIBS} )
