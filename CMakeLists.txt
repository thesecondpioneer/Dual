cmake_minimum_required(VERSION 3.22.1)
project(Dual)

set(CMAKE_CXX_STANDARD 23)
find_package(Ceres REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_executable(Dual main.cpp Dual.h
        Clock.h
)
target_link_libraries(Dual Ceres::ceres)
target_link_libraries(Dual Eigen3::Eigen)