#!/bin/bash
task_config=lerobot_liftobj_visual

python lwlab/scripts/env_server.py \
    --task_config="$task_config" \
    --headless
