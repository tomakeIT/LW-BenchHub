# Dockerfile for CI user image based on lwlab:1.6

FROM harbor.lightwheel.net/robot/lwlab:2.1

WORKDIR /workspace
RUN apt-get install -y libjemalloc2
ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"

WORKDIR /workspace/lwlab