# Dockerfile for the base image of lwlab

# 1. Base Image
# Use the OFFICIAL Isaac Sim image as the foundation.
# This inherits all necessary drivers, system libraries (like Vulkan, libGLU),
# and configurations, which resolves the segmentation fault and driver issues.
FROM harbor.lightwheel.net/robot/isaac-sim:4.5.0

# 2. Set up Proxy
# Pass proxy settings during the build.
ARG PROXY_HOST
ARG PROXY_PORT
ENV _http_proxy="http://${PROXY_HOST}:${PROXY_PORT}"
ENV _https_proxy="http://${PROXY_HOST}:${PROXY_PORT}"

# 3. Install System Dependencies for our own tools
# The base Isaac Sim image has most system deps, but we might need a few for ourselves.
# We'll install wget, git, and the build tools required by our Python packages.
# We override the original entrypoint to run our setup commands.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs wget curl sudo build-essential cmake pkg-config \
    # GUI and display libraries
    x11-apps mesa-utils libgl1-mesa-glx libgl1-mesa-dri libegl1-mesa \
    libx11-6 libxext6 libxrender1 libxrandr2 libxfixes3 libxi6 \
    # SSH and other tools
    openssh-server htop nvtop\
    linux-headers-$(uname -r) 
    # && rm -rf /var/lib/apt/lists/*

ENV PATH=/isaac-sim/kit/python/bin:$PATH

RUN /isaac-sim/kit/python/bin/python3 -m pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple/ && \
    /isaac-sim/kit/python/bin/python3 -m pip install --upgrade pip && \
    /isaac-sim/kit/python/bin/python3 -m pip config set global.extra-index-url https://artifactory.lightwheel.net/pip



# 4. Install Miniconda
# We install our own conda to manage Python environments without interfering
# with the system Python used by Isaac Sim's core components.
# ENV CONDA_DIR /opt/conda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
#     rm ~/miniconda.sh
# ENV PATH=$CONDA_DIR/bin:$PATH

# Create and activate a new conda environment
# RUN conda create -n lwlab python=3.10 -y
# SHELL ["conda", "run", "-n", "lwlab", "/bin/bash", "-c"]

# Upgrade pip to the latest version to avoid potential bugs in the default one.
# RUN pip install --upgrade pip

# 5. Install Base Python Packages from requirements files
# Copy our split requirements files into the image.
# COPY base-requirements-1.txt base-requirements-2.txt base-requirements-3.txt ./

# Install python packages in separate layers for better caching.
# Note: The isaac-sim base image might already contain some of these.
# Pip will handle existing packages gracefully.
# RUN pip install --no-cache-dir -r base-requirements-1.txt
# RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r base-requirements-2.txt
# RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r base-requirements-3.txt

# 6. Final Setup
# Set the final working directory.
# We do NOT set an ENTRYPOINT here. The final application Dockerfile will define it,
# effectively overriding the default Isaac Sim startup command.
RUN https_proxy=http://10.10.11.36:7897 http_proxy=http://10.10.11.36:7897 all_proxy=socks5://10.10.11.36:7897 /isaac-sim/warmup.sh --allow-root || true

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV OMNI_KIT_ALLOW_ROOT=1
USER root
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libglib2.0-0 \
    ncurses-term \
    wget \
    ffmpeg
COPY . /workspace/lwlab

# Set up a symbolic link between the installed Isaac Sim root folder and _isaac_sim in the Isaac Lab directory
RUN ln -sf /isaac-sim /workspace/lwlab/third_party/IsaacLab/_isaac_sim

# Install toml dependency
RUN /workspace/lwlab/third_party/IsaacLab/isaaclab.sh -p -m pip install toml

# Install apt dependencies for extensions that declare them in their extension.toml
RUN --mount=type=cache,target=/var/cache/apt \
    /workspace/lwlab/third_party/IsaacLab/isaaclab.sh -p /workspace/lwlab/third_party/IsaacLab/tools/install_deps.py apt /workspace/lwlab/third_party/IsaacLab/source && \
    apt -y autoremove && apt clean autoclean

# for singularity usage, have to create the directories that will binded
RUN mkdir -p /isaac-sim/kit/cache && \
    mkdir -p /root/.cache/ov && \
    mkdir -p /root/.cache/pip && \
    mkdir -p /root/.cache/nvidia/GLCache &&  \
    mkdir -p /root/.nv/ComputeCache && \
    mkdir -p /root/.nvidia-omniverse/logs && \
    mkdir -p /root/.local/share/ov/data && \
    mkdir -p /root/Documents

# Create SSH directory and copy SSH keys
RUN mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh
COPY docker/.ssh/id_rsa /root/.ssh/id_rsa
COPY docker/.ssh/id_rsa.pub /root/.ssh/id_rsa.pub
COPY docker/.ssh/known_hosts /root/.ssh/known_hosts
RUN chmod 600 /root/.ssh/id_rsa && \
    chmod 644 /root/.ssh/id_rsa.pub && \
    chmod 644 /root/.ssh/known_hosts

# for singularity usage, create NVIDIA binary placeholders
RUN touch /bin/nvidia-smi && \
    touch /bin/nvidia-debugdump && \
    touch /bin/nvidia-persistenced && \
    touch /bin/nvidia-cuda-mps-control && \
    touch /bin/nvidia-cuda-mps-server && \
    touch /etc/localtime && \
    mkdir -p /var/run/nvidia-persistenced && \
    touch /var/run/nvidia-persistenced/socket

# installing Isaac Lab dependencies
# use pip caching to avoid reinstalling large packages
RUN --mount=type=cache,target=/root/.cache/pip \
    /workspace/lwlab/third_party/IsaacLab/isaaclab.sh --install

# HACK: Remove install of quadprog dependency
RUN /workspace/lwlab/third_party/IsaacLab/isaaclab.sh -p -m pip uninstall -y quadprog

# aliasing isaaclab.sh and python for convenience
RUN echo "export ISAACLAB_PATH=/workspace/lwlab/third_party/IsaacLab" >> /root/.bashrc && \
    echo "alias isaaclab=/workspace/lwlab/third_party/IsaacLab/isaaclab.sh" >> /root/.bashrc && \
    echo "alias python=/workspace/lwlab/third_party/IsaacLab/_isaac_sim/python.sh" >> /root/.bashrc && \
    echo "alias python3=/workspace/lwlab/third_party/IsaacLab/_isaac_sim/python.sh" >> /root/.bashrc && \
    echo "alias pip='/workspace/lwlab/third_party/IsaacLab/_isaac_sim/python.sh -m pip'" >> /root/.bashrc && \
    echo "alias pip3='/workspace/lwlab/third_party/IsaacLab/_isaac_sim/python.sh -m pip'" >> /root/.bashrc && \
    echo "alias tensorboard='/workspace/lwlab/third_party/IsaacLab/_isaac_sim/python.sh /workspace/lwlab/third_party/IsaacLab/_isaac_sim/tensorboard'" >> /root/.bashrc && \
    echo "export TZ=$(date +%Z)" >> /root/.bashrc && \
    echo "shopt -s histappend" >> /root/.bashrc && \
    echo "PROMPT_COMMAND='history -a'" >> /root/.bashrc



WORKDIR /workspace