cmake_minimum_required(VERSION 3.16)
project(BingCropperSingle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc saliency)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found.")
endif()

message(STATUS "OpenCV version for BingCropperSingle: ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} bing_cropper_single.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if (CMAKE_VERSION VERSION_LESS "3.12")
    target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
else()
    set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
endif()

if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
    target_link_libraries(${PROJECT_NAME} PRIVATE "-lc++fs")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0 AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.1)
    target_link_libraries(${PROJECT_NAME} PRIVATE "-lstdc++fs")
endif()