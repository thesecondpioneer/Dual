cmake_minimum_required(VERSION 3.10)
project(Dual)

find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
set(CMAKE_BUILD_TYPE "RELEASE")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")
add_library(Dual INTERFACE DualDirectional.h)

add_executable(gradient_tests
        tests_grad/test_bennett5.cpp
        tests_grad/test_boxbod.cpp
        tests_grad/test_eckerle4.cpp
        tests_grad/test_mgh09.cpp
        tests_grad/test_mgh10.cpp
        tests_grad/test_rat42.cpp
        tests_grad/test_rat43.cpp
        tests_grad/test_thurber.cpp
        tests_grad/main.cpp
)

add_executable(hessian_tests
        tests_hess/test_mgh09.cpp
        tests_hess/main.cpp
)

target_include_directories(Dual INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(Dual INTERFACE Eigen3::Eigen)
target_link_libraries(gradient_tests PRIVATE Dual PUBLIC GTest::GTest GTest::Main)
target_link_libraries(hessian_tests PRIVATE Dual PUBLIC GTest::GTest GTest::Main)

include(GoogleTest)
gtest_discover_tests(gradient_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
gtest_discover_tests(hessian_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
