# Use base image that matches server CUDA version (12.8) and includes full development toolchain
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    git-lfs \
    ca-certificates \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
WORKDIR /tmp
RUN wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && /opt/conda/bin/conda clean -ya
ENV PATH="/opt/conda/bin:$PATH"

# Accept Conda terms and create the target environment
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create -n flashvsr python=3.11.13 -y

# Subsequent commands run inside the flashvsr environment
SHELL ["conda", "run", "-n", "flashvsr", "/bin/bash", "-c"]

# Set workdir and copy project (model weights are excluded by .dockerignore)
WORKDIR /workspace/FlashVSR-Pro
COPY . .

# Create optional directories for mounted models and inputs
RUN mkdir -p /workspace/FlashVSR-Pro/models/FlashVSR \
    && mkdir -p /workspace/FlashVSR-Pro/models/FlashVSR-v1.1 \
    && mkdir -p /workspace/FlashVSR-Pro/inputs

# 1. Install specific PyTorch build (important)
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 2. Install FlashVSR core dependencies
RUN pip install -e . \
    && pip install -r requirements.txt modelscope

# 3. Build and install Block-Sparse-Attention
WORKDIR /workspace/FlashVSR-Pro/Block-Sparse-Attention
RUN pip install packaging ninja
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
RUN python setup.py install

# # 4. Critical fix: patch block_sparse_attn package import issue (optional/manual step)
# RUN cd /opt/conda/envs/flashvsr/lib/python3.11/site-packages/block_sparse_attn && \
#     cat > __init__.py << 'EOF'
# __version__ = "0.0.1"
#
# from block_sparse_attn.block_sparse_attn_interface import (
#     block_sparse_attn_func,
#     block_streaming_attn_func,
#     token_streaming_attn_func
# )
#
# from . import utils
# EOF

# Clean caches to reduce image size
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    conda clean -ya && \
    pip cache purge

# 5. Initialize git-lfs (prepare for runtime model downloads)
WORKDIR /workspace/FlashVSR-Pro
RUN git lfs install

# Ensure entrypoint is executable (defensive)
RUN chmod +x entrypoint.sh

# Set container entrypoint
ENTRYPOINT ["entrypoint.sh"]
# Default to interactive bash if no command provided
CMD ["/bin/bash"]