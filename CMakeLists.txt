cmake_minimum_required(VERSION 3.10)

# Set the project name
project(FFTProject)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(FFTProject
        FFT/src/main.cpp
        FFT/src/Cooley-Tukey.cpp
        FFT/src/Cooley-Tukey-parallel.cpp)

# Include directories
include_directories(FFT/include)

# Link libraries if needed (e.g., pthread for parallel execution)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
target_link_libraries(FFTProject Threads::Threads)

# Set compiler flags
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")