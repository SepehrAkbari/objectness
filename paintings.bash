#!/usr/bin/env bash
set -e 

IMAGE_PATH="$1"

if [ -n "$IMAGE_PATH" ]; then
    if [ -d "$IMAGE_PATH" ]; then
        IMAGE_PATH="$(cd "$IMAGE_PATH" && pwd)"
    else
        echo "Error: Provided path '$IMAGE_PATH' is not a valid directory."
        exit 1
    fi
fi

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: '$1' is not installed or not in PATH."
        if [ -n "$2" ]; then echo "Fix: $2"; fi
        exit 1
    fi
}

check_command "git" "Git is not installed."
check_command "go" "Go is not installed."
check_command "cmake" "CMake is not installed."
check_command "uv" "Astral uv is not installed."

echo "All dependencies met."
echo ""

REPO_NAME="objectness"
REPO_URL="https://github.com/sepehrakbari/objectness.git"

if git rev-parse --is-inside-work-tree &> /dev/null && [ "$(basename "$(pwd)")" = "$REPO_NAME" ]; then
    echo "Already inside '$REPO_NAME'."
elif [ ! -d "$REPO_NAME" ]; then
    git clone "$REPO_URL"
    cd "$REPO_NAME"
else
    echo "Already exists. Navigating to '$REPO_NAME'."
    cd "$REPO_NAME"
fi

echo ""
if [ ! -f "go.mod" ]; then
    go mod init objectness
fi
go mod tidy
echo "Go module ready."

echo ""
if [ ! -d ".venv" ]; then
    uv venv
fi
uv pip install -r requirements.txt
echo "Python dependencies installed."

echo ""
mkdir -p main/bing_processor/build
cd main/bing_processor/build

rm -f CMakeCache.txt
cmake ../src
cmake --build .
cd ../../..
echo "BING compiled successfully."

echo ""
cd main
go build orchestrator.go
cd ..

echo ""

open_folder() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "$1"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        start "" "$1"
    elif command -v xdg-open &> /dev/null; then
        xdg-open "$1"
    fi
}

if [ -n "$IMAGE_PATH" ]; then
    echo "Using image path: $IMAGE_PATH"
    echo ""
    cd main
    ./orchestrator -data "$IMAGE_PATH"
else
    IMAGE_DIR="data/paintings"
    mkdir -p "$IMAGE_DIR"
    
    if [ -z "$(ls -A "$IMAGE_DIR" 2>/dev/null)" ]; then
        echo "Waiting for images..."
        
        open_folder "$PWD/$IMAGE_DIR" || true
        
        read -p "Drag and drop images into the folder window, then press [Enter] to start processing..."
    fi
    
    echo ""

    cd main
    ./orchestrator -data "../$IMAGE_DIR"
fi