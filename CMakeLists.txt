cmake_minimum_required(VERSION 3.10)
project(voxl-aruco-detector CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV4 REQUIRED opencv4)

include_directories(
  ${OPENCV4_INCLUDE_DIRS}
  include
)

link_directories(${OPENCV4_LIBRARY_DIRS})

add_executable(voxl-aruco-detector
  src/aruco_publisher.cpp
)

target_link_libraries(voxl-aruco-detector
  ${OPENCV4_LIBRARIES}
  modal_pipe
  voxl_cutils
)
