#!/bin/bash

WORK_DIR=$(pwd)
IMAGE_DIR=image_compression/images
PYTHON_SCRIPT=full_analysis.py
EXECUTABLE=image_compression/build/image_compression_exec

echo "Running project locally without Docker..."

# 1. Check for images
echo "Available images:"
IMAGE_LIST=($(ls $IMAGE_DIR))

if [ ${#IMAGE_LIST[@]} -eq 0 ]; then
    echo "No images found in the $IMAGE_DIR directory!"
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

# 2. Rebuild the project
echo "Rebuilding the C++ project..."
cd image_compression
rm -rf build
mkdir build
cd build
cmake ..
make

if [ $? -ne 0 ]; then
    echo "Build failed! Exiting."
    exit 1
fi
cd ../..

echo "Build completed."

# 3. Check or create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install numpy matplotlib
else
    echo "Virtual environment found. Activating..."
    source .venv/bin/activate
fi

# 4. Run the C++ program on the selected image
echo "Running the C++ program on the selected image..."
$EXECUTABLE $IMAGE_DIR/$IMAGE_FILE

# 5. Run the Python full analysis
echo "Running the Python analysis..."
cd Python_Plot
python3 $PYTHON_SCRIPT

echo "All tasks completed successfully."