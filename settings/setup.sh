PROJECT_DIR=$(pwd)

# Build hints for CUDA extensions and Torch CMake paths (best-effort, safe if CUDA not present)
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CPATH=${CPATH:-$CUDA_HOME/include:$CPATH}
TORCH_CMAKE_PREFIX=$(python3 -c "import torch; import os; print(getattr(torch.utils, 'cmake_prefix_path', ''))" 2>/dev/null || true)
if [ -n "$TORCH_CMAKE_PREFIX" ]; then
	export CMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
fi

# Pytorch
pip3 install -i https://download.pytorch.org/whl/cu121 -U torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
pip3 install -i https://download.pytorch.org/whl/cu121 -U xformers==0.0.27

# A modified gaussian splatting (+ alpha, depth, normal rendering)
cd extensions && git clone https://github.com/BaowenZ/RaDe-GS.git --recursive --depth 1 && cd RaDe-GS/submodules
# Prefer current env for CUDA extensions to avoid resolver swapping
if [ -d diff-gaussian-rasterization ]; then
	pip3 install --no-build-isolation ./diff-gaussian-rasterization || pip3 install ./diff-gaussian-rasterization
fi
if [ -d simple-knn ]; then
	pip3 install --no-build-isolation ./simple-knn || true
fi
cd ${PROJECT_DIR}

# Others
pip3 install -U gpustat
pip3 install -U -r settings/requirements.txt
# Install ffmpeg if available and permitted (do not fail on environments without sudo)
if command -v apt-get >/dev/null 2>&1; then
	if command -v sudo >/dev/null 2>&1; then
		sudo apt-get update -y && sudo apt-get install -y ffmpeg || true
	else
		apt-get update -y && apt-get install -y ffmpeg || true
	fi
fi
