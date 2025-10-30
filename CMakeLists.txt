cmake_minimum_required(VERSION 3.10)
project(voxl-aruco-detector CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
find_package(OpenCV QUIET)

# 見つからない場合は pkg-config で opencv4 → opencv の順に探す
if(NOT OpenCV_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(OPENCV4 QUIET opencv4)
  if(OPENCV4_FOUND)
    set(OpenCV_INCLUDE_DIRS ${OPENCV4_INCLUDE_DIRS})
    set(OpenCV_LIBRARIES    ${OPENCV4_LIBRARIES})
  else()
    pkg_check_modules(OPENCV REQUIRED opencv)   # ここで 'opencv' を使う
    set(OpenCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIRS})
    set(OpenCV_LIBRARIES    ${OPENCV_LIBRARIES})
  endif()
endif()


pkg_check_modules(OPENCV4 REQUIRED opencv4)

include_directories(${OpenCV_INCLUDE_DIRS} include)
add_executable(voxl-aruco-detector src/aruco_publisher.cpp)
target_link_libraries(voxl-aruco-detector ${OpenCV_LIBRARIES} modal_pipe voxl_cutils)
