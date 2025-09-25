#!/bin/bash

task_config=g1-controller
export LW_API_ENDPOINT="https://api-dev.lightwheel.net"
python ./lwlab/scripts/teleop/teleop_main.py \
    --task_config="$task_config"
