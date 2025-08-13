# Application Dockerfile for lwlab
# This Dockerfile builds the final application image on top of 'lwlab:base'.

# 1. Base Image
# Start from the pre-built base image which contains all dependencies.
FROM lwlab:base


# 2. Copy Local Project Code
# Copy the entire local 'lwlab' directory into the image.
WORKDIR /workspace
# COPY . /workspace/lwlab

# WORKDIR /workspace/lwlab/third_party/robosuite
# RUN pip install -e .

WORKDIR /workspace/lwlab/third_party/robocasa
RUN pip install -e .

WORKDIR /workspace/lwlab
RUN pip install -e .

RUN python3 -m pip install numpy==1.26.4
RUN python3 -m pip install onnxruntime

# 5. Final Setup
WORKDIR /workspace/lwlab

# RUN https_proxy=http://10.10.11.36:7897 http_proxy=http://10.10.11.36:7897 all_proxy=socks5://10.10.11.36:7897 /isaac-sim/warmup.sh || true

# Entrypoint to set PYTHONPATH, activate conda, and start bash.
# We prepend the official isaacsim package path to PYTHONPATH.
# This ensures that our custom 'lwlab' conda environment can find and use
# the 'isaacsim' module from the base NVIDIA image, solving the ModuleNotFoundError.
ENTRYPOINT ["/bin/bash", "-c", "-i"] 