cmake_minimum_required(VERSION 3.10)
project(Dual)

find_package(Eigen3 REQUIRED)

add_library(Dual INTERFACE)
add_executable(tests tests/test_mgh09.cpp tests/test_thurber.cpp)

target_include_directories(Dual INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Dual.h Clock.h jet.h SparseDual.h HyperDual.h HyperDual.h HyperDual.h DualScalar.h

target_link_libraries(Dual INTERFACE Eigen3::Eigen)
target_link_libraries(tests PRIVATE Dual)
