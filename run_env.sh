#!/bin/bash

task_config=g1-coffee

python ./lwlab/scripts/env_server.py \
    --task_config="$task_config" \
    --headless
