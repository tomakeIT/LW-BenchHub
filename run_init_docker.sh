#!/usr/bin/env bash
set -e -o pipefail

# Docker initialization script to pull all branches and switch to yong.jin/uploader
# Usage: ./run_init_docker.sh [docker_tag]

echo "=== Docker Initialization Script ==="

# Parse the command line arguments
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_TAG=${1:-"1.4"}
CONTAINER_NAME="lwlab_${DOCKER_TAG}"
TARGET_BRANCH="yong.jin/uploader"

echo "Docker Tag: $DOCKER_TAG"
echo "Container Name: $CONTAINER_NAME"
echo "Target Branch: $TARGET_BRANCH"

# Check if container exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "=== Container exists, checking status ==="
    
    # Container exists, check if it's running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container is already running, initializing git..."
    else
        echo "Container exists but not running, starting it..."
        docker start ${CONTAINER_NAME}
    fi
    
    # # Execute git initialization in the existing container
    # echo "=== Initializing Git Repository ==="
    # docker exec -it ${CONTAINER_NAME} /bin/bash -c "
    #     cd /workspace/lwlab && \
    #     echo 'Current directory: '\$(pwd) && \
    #     if [ ! -d '.git' ]; then \
    #         echo 'Error: /workspace/lwlab is not a git repository'; \
    #         exit 1; \
    #     fi && \
    #     echo '=== Fetching all branches ===' && \
    #     git fetch --all && \
    #     echo '=== Available remote branches ===' && \
    #     git branch -r && \
    #     echo '=== Switching to branch ${TARGET_BRANCH} ===' && \
    #     if git branch --list '${TARGET_BRANCH}' | grep -q '${TARGET_BRANCH}'; then \
    #         echo 'Local branch ${TARGET_BRANCH} exists, checking out...'; \
    #         git checkout '${TARGET_BRANCH}'; \
    #     else \
    #         echo 'Creating and checking out local branch ${TARGET_BRANCH} from origin...'; \
    #         git checkout -b '${TARGET_BRANCH}' origin/'${TARGET_BRANCH}'; \
    #     fi && \
    #     echo '=== Pulling latest changes ===' && \
    #     git pull origin '${TARGET_BRANCH}' && \
    #     echo '=== Current branch and status ===' && \
    #     git branch --show-current && \
    #     git status --short && \
    #     echo '=== Cloning robocasa_upload repository ===' && \
    #     cd third_party/ && \
    #     if [ -d 'robocasa_upload' ]; then \
    #         echo 'robocasa_upload directory already exists, updating...' && \
    #         cd robocasa_upload && \
    #         git fetch --all; \
    #     else \
    #         echo 'Cloning robocasa_upload repository...' && \
    #         git clone ssh://git@git.lightwheel.ai:2022/robot/robocasa/robocasa_upload.git && \
    #         cd robocasa_upload; \
    #     fi && \
    #     echo '=== Switching to yong.jin/dev branch ===' && \
    #     if git branch --list 'yong.jin/dev' | grep -q 'yong.jin/dev'; then \
    #         echo 'Local branch yong.jin/dev exists, checking out...' && \
    #         git checkout 'yong.jin/dev'; \
    #     else \
    #         echo 'Creating and checking out local branch yong.jin/dev from origin...' && \
    #         git checkout -b 'yong.jin/dev' origin/'yong.jin/dev'; \
    #     fi && \
    #     echo '=== Pulling latest changes for robocasa_upload ===' && \
    #     git pull origin 'yong.jin/dev' && \
    #     echo '=== robocasa_upload setup completed ===' && \
    #     cd /workspace/lwlab && \
    #     echo '=== Git initialization completed ==='
    # "
    
    # Enter interactive mode
    echo "=== Entering interactive mode ==="
    docker exec -it ${CONTAINER_NAME} bash
    
else
    echo "=== Container doesn't exist, creating new one ==="
    # Container doesn't exist, create and run new one
    # Setup X11 forwarding for GUI applications
    echo "=== Setting up X11 forwarding ==="
    
    # Create xauth file for secure X11 forwarding
    XAUTH_FILE="/tmp/.docker.xauth-${USER}"
    if [ ! -f "${XAUTH_FILE}" ]; then
        touch "${XAUTH_FILE}"
        chmod 644 "${XAUTH_FILE}"
        xauth nlist "${DISPLAY}" | sed -e 's/^..../ffff/' | xauth -f "${XAUTH_FILE}" nmerge - 2>/dev/null || true
    fi
    
    # Grant X11 access (more secure than xhost +)
    xhost +local:root > /dev/null 2>&1 || echo "Warning: xhost command failed, X11 forwarding may not work"
    
    # First, create container in detached mode with X11 support
    docker run \
        -itd \
        --name ${CONTAINER_NAME} \
        --gpus all \
        --privileged \
        -e DISPLAY=${DISPLAY} \
        -e QT_X11_NO_MITSHM=1 \
        -e XAUTHORITY=/tmp/.docker.xauth \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v "${XAUTH_FILE}:/tmp/.docker.xauth:rw" \
        -v /data/ceph_hdd:/data/ceph_hdd \
        -v /lw_cache/lwlab/isaac-cache/isaac-sim_kit_cache:/isaac-sim/kit/cache \
        -v /lw_cache/lwlab/isaac-cache/cache_ov:/root/.cache/ov \
        -v /lw_cache/lwlab/isaac-cache/lwlab:/root/.cache/lwlab \
        -v /lw_cache/lwlab/output:/output \
        --user root \
        harbor.lightwheel.net/robot/lwlab:${DOCKER_TAG} \
        bash
    
    # Wait a moment for container to be ready
    sleep 2
    
    # Execute git initialization in the container
    echo "=== Initializing Git Repository ==="
    docker exec ${CONTAINER_NAME} bash -c "
        cd /workspace/lwlab
        echo 'Current directory: '\$(pwd)
        
        if [ ! -d '.git' ]; then
            echo 'Error: /workspace/lwlab is not a git repository'
            exit 1
        fi
        
        echo '=== Fetching all branches ==='
        git fetch --all
        
        echo '=== Available remote branches ==='
        git branch -r
        
        echo '=== Switching to branch ${TARGET_BRANCH} ==='
        if git branch --list '${TARGET_BRANCH}' | grep -q '${TARGET_BRANCH}'; then
            echo 'Local branch ${TARGET_BRANCH} exists, checking out...'
            git checkout '${TARGET_BRANCH}'
        else
            echo 'Creating and checking out local branch ${TARGET_BRANCH} from origin...'
            git checkout -b '${TARGET_BRANCH}' origin/'${TARGET_BRANCH}'
        fi
        
        echo '=== Pulling latest changes ==='
        git pull origin '${TARGET_BRANCH}'
        
        echo '=== Current branch and status ==='
        git branch --show-current
        git status --short
        
        echo '=== Cloning robocasa_upload repository ==='
        cd third_party/
        if [ -d 'robocasa_upload' ]; then
            echo 'robocasa_upload directory already exists, updating...'
            cd robocasa_upload
            git fetch --all
        else
            echo 'Cloning robocasa_upload repository...'
            git clone ssh://git@git.lightwheel.ai:2022/robot/robocasa/robocasa_upload.git
            cd robocasa_upload
        fi
        
        echo '=== Switching to yong.jin/dev branch ==='
        if git branch --list 'yong.jin/dev' | grep -q 'yong.jin/dev'; then
            echo 'Local branch yong.jin/dev exists, checking out...'
            git checkout 'yong.jin/dev'
        else
            echo 'Creating and checking out local branch yong.jin/dev from origin...'
            git checkout -b 'yong.jin/dev' origin/'yong.jin/dev'
        fi
        
        echo '=== Pulling latest changes for robocasa_upload ==='
        git pull origin 'yong.jin/dev'

        pip install -e .
        
        echo '=== robocasa_upload setup completed ==='
        cd /workspace/lwlab
        
        echo '=== Git initialization completed ==='
    "
    
    # Enter interactive mode
    echo "=== Entering interactive mode ==="
    docker exec -it ${CONTAINER_NAME} bash
fi

echo "=== Docker initialization completed ==="

# Cleanup function to revoke X11 access when script exits
cleanup() {
    echo "=== Cleaning up X11 permissions ==="
    xhost -local:root > /dev/null 2>&1 || true
    # Clean up xauth file if it exists
    if [ -f "/tmp/.docker.xauth-${USER}" ]; then
        rm -f "/tmp/.docker.xauth-${USER}" 2>/dev/null || true
    fi
}

# Set trap to call cleanup function on script exit
trap cleanup EXIT
