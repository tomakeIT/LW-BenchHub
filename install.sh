pip install --upgrade pip
pip install uv
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
conda install pinocchio -c conda-forge -y

# install isaacsim
# uv pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
source /home/jialeng/Desktop/lw_benchhub/third_party/IsaacLab-Arena/submodules/IsaacLab/_isaac_sim/setup_conda_env.sh

# git submodule update --init --recursive

# install isaaclab
cd third_party/IsaacLab-Arena/submodules/IsaacLab
bash isaaclab.sh --install

# install isaaclab-arena
cd ../..
uv pip install -e .

# install lw_benchhub
cd ../..
uv pip install -e .
# If you need to use LeRobot in LW-BenchHub, install it's requirements by
# uv pip install -e .[lerobot]

