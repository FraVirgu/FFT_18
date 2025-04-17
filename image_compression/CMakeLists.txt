cmake_minimum_required(VERSION 3.10)

# Project name
project(image_compression_exec)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find packages before using their variables
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenCV REQUIRED)

# Add the executable
add_executable(image_compression_exec
        ../FFT/src/ParallelIterativeFFT.cpp
        ../FFT/src/Cooley-Tukey.cpp
        src/fft_analysis.cpp
        src/svd_analysis.cpp
        src/error_plot.cpp
        src/fft2d.cpp
        #src/fft2d_par.cpp
        image_main.cpp
)

# Include directories
include_directories(
        ../FFT/include
        include
        ${MPI_INCLUDE_PATH}
        ${OpenCV_INCLUDE_DIRS}
)



target_link_libraries(image_compression_exec
        ${MPI_LIBRARIES}
        OpenMP::OpenMP_CXX
        Threads::Threads
        ${OpenCV_LIBS}
        curl
        sqlite3
)