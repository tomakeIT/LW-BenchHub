#!/bin/bash
task_config=lerobot_liftobj_state

python ./policy/maniskill_ppo/train.py \
    --task_config="$task_config" \
    --headless \

