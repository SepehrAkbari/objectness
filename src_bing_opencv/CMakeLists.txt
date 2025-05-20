cmake_minimum_required(VERSION 3.16) # Increased for better modern CMake practices
project(PaintingCropperBING LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF) # Good practice

# Find OpenCV package
# For Homebrew on Apple Silicon, OpenCV might be in /opt/homebrew
# You can set OpenCV_DIR if CMake doesn't find it automatically:
# set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4") # Adjust path if necessary
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc saliency)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found! Please ensure it's installed and CMake can find it. You might need to set OpenCV_DIR.")
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main_cropper.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# Optional: For better executable location after build
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})