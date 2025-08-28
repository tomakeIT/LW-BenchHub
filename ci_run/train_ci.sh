#!/bin/bash
task_config=ci_default
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

rm -rf /workspace/lwlab/policy/skrl/logs/*

python3 /workspace/lwlab/lwlab/scripts/rl/train.py \
    --task_config="$task_config" \
    --headless 2>&1


PYTHON_EXIT_CODE=$?

pt_found=0
pt_file=$(find /workspace/lwlab/policy/skrl/logs/ -type f -path "*/checkpoints/*.pt" | head -n 1)
if [ -n "$pt_file" ]; then
    pt_found=1
fi

if [ ${PYTHON_EXIT_CODE} -eq 0 ] && [ $pt_found -eq 1 ]; then
    python3 -c "
import json
with open('${OUTPUT_JSON}', 'w') as out_f:
    json.dump({'success': True, 'desc':'RL training success', 'error': None}, out_f, indent=4, ensure_ascii=False)
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
