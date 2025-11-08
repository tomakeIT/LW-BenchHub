FROM harbor.lightwheel.net/robot/lwlab:base_isaaclab-fb270ab5_gr00t-3bce5530_lerobot_1108


ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV CONDA_DIR=/opt/conda
ENV ENV_NAME=lwlab
ENV PATH="$CONDA_DIR/bin:$CONDA_DIR/envs/$ENV_NAME/bin:$PATH"
# proxy from arg
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}


### local current dir is lwlab/ (with submodules), and add ./third_party/robocasa_upload
# Copy all directories to /workspace/lwlab
RUN rm -rf /workspace/lwlab
COPY . /workspace/lwlab/
RUN rm -rf /workspace/lwlab/docker

# install robocasa_upload
WORKDIR /workspace/lwlab/third_party/robocasa_upload
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    pip install -e .

# install lwlab
WORKDIR /workspace/lwlab
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    pip install -e .

# install IsaacLab-Arena
WORKDIR /workspace/lwlab/third_party/IsaacLab-Arena
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    pip install -e .

# clear proxy
ENV HTTP_PROXY=
ENV HTTPS_PROXY=
RUN unset HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

WORKDIR /workspace/lwlab

ENTRYPOINT ["/bin/bash", "-c", "-i"]