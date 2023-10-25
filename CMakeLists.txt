cmake_minimum_required(VERSION 3.22.1)
project(Dual)

set(CMAKE_CXX_STANDARD 17)
find_package(Ceres REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

add_executable(Dual main.cpp Dual.h
        Clock.h
)
target_link_libraries(Dual Ceres::ceres)
target_link_libraries(Dual Eigen3::Eigen)