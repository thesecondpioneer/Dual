cmake_minimum_required(VERSION 3.10)
project(Dual)

set(CMAKE_CXX_STANDARD 17)
#find_package(Ceres REQUIRED)
#find_package(Eigen3 3.4 REQUIRED NO_MODULE)

 #set(CMAKE_SYSTEM_NAME Linux)
 #set(CMAKE_SYSTEM_PROCESSOR arm)

 #set(SDK /home/mdc/sdk/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu)
 #set(CMAKE_C_COMPILER ${SDK}/bin/aarch64-none-linux-gnu-gcc)
 #set(CMAKE_CXX_COMPILER ${SDK}/bin/aarch64-none-linux-gnu-g++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

add_executable(Dual main.cpp Dual.h
        Clock.h
        jet.h
)
#target_link_libraries(Dual Ceres::ceres)
#target_link_libraries(Dual Eigen3::Eigen)
