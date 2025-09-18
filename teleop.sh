#!/bin/bash

task_config=g1-controller
export LW_API_ENDPOINT="http://usdcache-dev.lightwheel.net:30807"
python ./lwlab/scripts/teleop/teleop_main.py \
    --task_config="$task_config"
