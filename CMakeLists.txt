cmake_minimum_required(VERSION 3.5)
project(C++_ROBOTICS)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(Eigen3 REQUIRED)

include_directories(./include)
set(EIGEN_ROOT_DIR "/usr/local/include/eigen3/")

include_directories(
    ${EIGEN_ROOT_DIR}
)


##############################
#Localization
##############################

add_executable(kf src/kalmanFilter.cpp)
# target_include_directories(kf PRIVATE ./include)
target_link_libraries(kf PRIVATE Eigen3::Eigen)