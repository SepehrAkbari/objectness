cmake_minimum_required(VERSION 3.16)
project(BingCropperSingle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Adjust this path if your OpenCV installation is not found automatically
# For Homebrew on Apple Silicon, it might be:
# set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc saliency)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found! Please ensure it's installed and CMake can find it. You might need to set OpenCV_DIR in CMakeLists.txt.")
endif()

message(STATUS "OpenCV version for BingCropperSingle: ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})

# Ensure the source file name matches what you save it as
add_executable(${PROJECT_NAME} bing_cropper_single.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# Output executable directly into the build directory relative to this CMakeLists.txt
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Ensure C++17 filesystem is available
if (CMAKE_VERSION VERSION_LESS "3.12")
    target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
else()
    set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
endif()

# Link against stdc++fs or c++fs if needed for older compilers/systems for std::filesystem
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
    target_link_libraries(${PROJECT_NAME} PRIVATE "-lc++fs")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0 AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.1)
    # GCC 8 needs -lstdc++fs, GCC 9.1+ has it in libstdc++ by default
    target_link_libraries(${PROJECT_NAME} PRIVATE "-lstdc++fs")
endif()