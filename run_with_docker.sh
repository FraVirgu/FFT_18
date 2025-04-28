#!/bin/bash

CONTAINER_NAME=hpc-projects
WORK_DIR=/home/ubuntu/shared-folder/AMSC_FFT
PYTHON_SCRIPT=full_analysis.py

echo "Starting Docker container ($CONTAINER_NAME)..."

docker start $CONTAINER_NAME > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Error: the container $CONTAINER_NAME does not exist. Please create it first!"
    exit 1
fi
echo "Container started."

echo "Available images:"
IMAGE_LIST=($(ls image_compression/images))

if [ ${#IMAGE_LIST[@]} -eq 0 ]; then
    echo "No images found in the images directory!"
    exit 1
fi

for i in "${!IMAGE_LIST[@]}"; do
    echo "$((i+1)). ${IMAGE_LIST[$i]}"
done

echo
while true; do
    echo "Select the number of the image you want to process:"
    read selection

    if [[ $selection =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le ${#IMAGE_LIST[@]} ]; then
        IMAGE_FILE=${IMAGE_LIST[$((selection-1))]}
        break
    else
        echo "Invalid selection, please try again."
    fi
done

echo "You selected: $IMAGE_FILE"

docker exec -it $CONTAINER_NAME bash -c "
    set -e

    echo 'Navigating to the project directory...'
    cd $WORK_DIR

    if [ ! -d '.venv' ]; then
        echo 'Creating virtual environment...'
        python3 -m venv .venv
        source .venv/bin/activate
        pip install numpy matplotlib --break-system-packages
    else
        echo 'Virtual environment .venv found.'
        source .venv/bin/activate
    fi

    echo 'Rebuilding the C++ project...'
    cd image_compression
    rm -rf build
    mkdir build
    cd build
    cmake ..
    make

    echo 'Running the C++ program on the selected image...'
    ./image_compression_exec ../images/$IMAGE_FILE

    echo 'Running the Python analysis...'
    cd ../../Python_Plot
    python3 $PYTHON_SCRIPT
"

echo "All tasks completed."