source /home/jialeng/Desktop/lw_benchhub/third_party/IsaacLab-Arena/submodules/IsaacLab/_isaac_sim/setup_conda_env.sh

# 设置 GPU 架构环境变量以支持 Blackwell (GB10) 架构
# GB10 的计算能力是 12.1，应该使用 sm_121 或 12.1+PTX
# 如果 sm_121 不工作，可以尝试以下选项：
# - "12.1+PTX" (使用 PTX 即时编译，更兼容)
# - "sm_121" (直接指定架构)
# - "sm_121+PTX" (同时支持 sm_121 和 PTX)
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"

export TORCH_CUDA_ARCH_LIST="12.1+PTX"

python3  /home/jialeng/Desktop/lw_benchhub/lw_benchhub/scripts/teleop/replay_demos.py   --dataset_file /home/jialeng/Desktop/LightwheelData/coffeeServeMug/CoffeeServeMug/CoffeeServeMug_1768453993358915/dataset_success.hdf5   --enable_cameras
python ./lw_benchhub/scripts/teleop/replay_action_demo.py \
--dataset_file /home/jialeng/Desktop/LightwheelData/coffeeServeMug/CoffeeServeMug/CoffeeServeMug_1768453993358915/dataset_success.hdf5 \
--replay_mode action \
--enable_cameras




python3 lw_benchhub/scripts/env_server.py --enable_camera



########################################################
# run with restful protocol
LW_REST_CLIENT_DEBUG=1 XLA_PYTHON_CLIENT_PREALLOCATE=false python3 lw_benchhub/scripts/policy/eval_policy.py \
  --config policy/PI/deploy_policy_x7s.yml \
  --remote_protocol restful \
  --server_host 127.0.0.1 \
  --server_port 8001 \
  --debug_step_interval 1 \
  --debug_client_flow

LW_REST_DEBUG=1 python3 lw_benchhub/scripts/env_server.py \
  --remote_protocol restful \
  --restful_host 0.0.0.0 \
  --restful_port 8001 \
  --enable_camera

########################################################
# run with ipc protocol
LW_IPC_CLIENT_DEBUG=1 XLA_PYTHON_CLIENT_PREALLOCATE=false python3 lw_benchhub/scripts/policy/eval_policy.py \
  --config policy/PI/deploy_policy_x7s.yml \
  --remote_protocol ipc \
  --server_host 127.0.0.1 \
  --server_port 50000 \
  --ipc_authkey lightwheel \
  --debug_step_interval 1 \
  --debug_client_flow

LW_IPC_DEBUG=1 python3 lw_benchhub/scripts/env_server.py \
  --remote_protocol ipc \
  --ipc_host 127.0.0.1 \
  --ipc_port 50000 \
  --ipc_authkey lightwheel \
  --enable_camera

