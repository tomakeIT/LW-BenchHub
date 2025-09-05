#!/bin/bash
task_config=$1
env_gpu=0
policy_gpu=0
export LW_API_ENDPOINT="http://api-dev.lightwheel.net:30807"
if [ "$env_gpu" -eq "$policy_gpu" ]; then
    export CUDA_VISIBLE_DEVICES=${env_gpu}
    export ENV_GPU=0
    export POLICY_GPU=0
else
    export CUDA_VISIBLE_DEVICES="${env_gpu},${policy_gpu}"
    export ENV_GPU=0
    export POLICY_GPU=1
fi

DATASET_DIR=/output
OUTPUT_JSON="${DATASET_DIR}/result.json"
pt_found=0
success=True
success_rate=0
if [ "$task_config" == "ci_lift_obj" ] || [ "$task_config" == "ci_lift_obj_daily" ]; then
    rm -rf /workspace/lwlab/policy/maniskill_ppo/logs/*
    python3 /workspace/lwlab/policy/maniskill_ppo/train.py \
        --task_config="$task_config" \
        --headless 2>&1

    pt_file=$(find /workspace/lwlab/policy/maniskill_ppo/logs/ -type f -path "*/*.pt" | head -n 1)
    if [ "$task_config" == "ci_lift_obj_daily" ]; then
        json_file=$(find /workspace/lwlab/policy/maniskill_ppo/logs/ -type f -path "*/result.json")
        if [ -n "$json_file" ]; then
            success_rate=$(python3 -c "import json; print(json.load(open('$json_file')).get('success_rate', ''))")
            success=$(python3 -c "import json; print(json.load(open('$json_file')).get('ci_success', ''))")
        else
            success=False
        fi
    fi
    
elif [ "$task_config" == "ci_open_draw" ]; then
    rm -rf /workspace/lwlab/policy/skrl/logs/*
    python3 /workspace/lwlab/lwlab/scripts/rl/train.py \
        --task_config="$task_config" \
        --headless 2>&1

    pt_file=$(find /workspace/lwlab/policy/skrl/logs/ -type f -path "*/checkpoints/*.pt" | head -n 1)
fi

PYTHON_EXIT_CODE=$?

if [ -n "$pt_file" ]; then
    pt_found=1
fi

if [ ${PYTHON_EXIT_CODE} -eq 0 ] && [ $pt_found -eq 1 ]; then
    python3 -c "
import json
with open('${OUTPUT_JSON}', 'w') as out_f:
    json.dump({'success': ${success}, 'desc':'RL training ${task_config}, only daily ci check success_rate: ${success_rate}', 'error': None}, out_f, indent=4, ensure_ascii=False)
"
else
    python3 -c "
import json
with open('${OUTPUT_JSON}', 'w') as f:
    json.dump({
        'success': False,
        'desc': 'RL training failed',
        'error': '${PYTHON_EXIT_CODE} or not success save pt file'
    }, f, indent=4, ensure_ascii=False)
"
    exit 1
fi
