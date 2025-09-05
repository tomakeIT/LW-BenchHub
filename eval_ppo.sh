#!/bin/bash

gpu_id=0

export CUDA_VISIBLE_DEVICES=${gpu_id}
export ENV_GPU=${gpu_id}
export POLICY_GPU=${gpu_id}
task_config=lerobot_liftobj_state_play

python ./policy/maniskill_ppo/play.py \
    --task_config="$task_config" \
    # --enable_cameras \
    # --headless \
