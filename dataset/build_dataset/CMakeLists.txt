cmake_minimum_required(VERSION 3.5)
project(BuildDataset)

set(CMAKE_CXX_STANDARD 14)


find_package(OpenCV 3 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()

add_executable(BuildDataset BuildDataset.cpp)
target_link_libraries(BuildDataset ${OpenCV_LIBS} )
