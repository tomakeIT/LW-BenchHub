#!/bin/bash

task_config=g1-coffee
export ENV_GPU=0
export POLICY_GPU=0
export LW_API_ENDPOINT="http://usdcache-dev.lightwheel.net:30807"
python ./lwlab/scripts/policy/env_serve.py \
    --task_config="$task_config" \
    --headless
