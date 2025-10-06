#!/bin/bash

task_config=pnp-orange
export LW_API_ENDPOINT="https://api-dev.lightwheel.net"
python ./lwlab/scripts/env_server.py \
    --task_config="$task_config" \
    --headless
