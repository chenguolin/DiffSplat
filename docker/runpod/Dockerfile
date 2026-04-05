# Optional Dockerfile for DiffSplat: GPU dev + services for RunPod
# Non-invasive: lives at repo root under docker/runpod/

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Core system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    nginx \
    curl wget ca-certificates \
    git \
    build-essential \
    python3 python3-pip python3-dev \
    # cuDNN 9 for CUDA 12 provider paths (ONNX/Torch)
    libcudnn9-cuda-12 \
 && rm -rf /var/lib/apt/lists/*

# Python: Torch cu121 + tools (pin to known-good)
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools && \
    pip3 install --no-cache-dir \
      torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir xformers==0.0.27 jupyterlab filebrowser-cli

# App deps if present
COPY requirements.txt /tmp/requirements.txt
RUN if [ -f /tmp/requirements.txt ]; then pip3 install --no-cache-dir -r /tmp/requirements.txt; fi

# SSH daemon setup
RUN mkdir -p /var/run/sshd /root/.ssh && \
    chmod 700 /root/.ssh

# Expose standard service ports
EXPOSE 22 80 9090 9999

# Add startup script
COPY docker/runpod/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
