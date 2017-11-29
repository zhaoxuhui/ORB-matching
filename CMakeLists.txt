cmake_minimum_required(VERSION 2.6)
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
project(match)

# OpenCV
find_package( OpenCV 3.1 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(match main.cpp)

target_link_libraries( match ${OpenCV_LIBS} )

install(TARGETS match RUNTIME DESTINATION bin)
