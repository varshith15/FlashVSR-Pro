# 使用与服务器CUDA版本(12.8)匹配的基础镜像（包含完整开发工具链）
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    git-lfs \
    ca-certificates \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 安装Miniconda
WORKDIR /tmp
RUN wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && /opt/conda/bin/conda clean -ya
ENV PATH="/opt/conda/bin:$PATH"

# 接受Conda服务条款并创建指定环境
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create -n flashvsr python=3.11.13 -y

# 后续命令在flashvsr环境中执行
SHELL ["conda", "run", "-n", "flashvsr", "/bin/bash", "-c"]

# 设置工作目录并复制项目代码（模型文件已被.dockerignore排除）
WORKDIR /workspace/FlashVSR-Pro
COPY . .

# 为将被挂载的模型和数据创建必要的空目录结构（可选但推荐）
RUN mkdir -p /workspace/FlashVSR/examples/WanVSR/FlashVSR \
    && mkdir -p /workspace/FlashVSR/examples/WanVSR/FlashVSR-v1.1 \
    && mkdir -p /workspace/FlashVSR/examples/WanVSR/inputs

# 1. 首先安装指定版本的PyTorch（关键步骤）
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 2. 安装FlashVSR核心依赖
RUN pip install -e . \
    && pip install -r requirements.txt modelscope \
    && pip cache purge

# 3. 编译并安装Block-Sparse-Attention
WORKDIR /workspace/FlashVSR/Block-Sparse-Attention
RUN pip install packaging ninja
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
RUN python setup.py install

# 4. 关键修复：修正block_sparse_attn包的__init__.py文件中的导入错误
RUN cd /opt/conda/envs/flashvsr/lib/python3.11/site-packages/block_sparse_attn && \
    cat > __init__.py << 'EOF'
__version__ = "0.0.1"

from block_sparse_attn.block_sparse_attn_interface import (
    block_sparse_attn_func,
    block_streaming_attn_func,
    token_streaming_attn_func
)

from . import utils
EOF

# 5. 初始化git-lfs（为运行时可能下载模型做准备）
WORKDIR /workspace/FlashVSR/examples/WanVSR
RUN git lfs install

# 确保脚本可执行（冗余操作，但更安全）
RUN chmod +x /entrypoint.sh

# 设置容器启动时的默认入口点
ENTRYPOINT ["/entrypoint.sh"]
# 如果没有提供其他命令，默认启动交互式 Bash
CMD ["/bin/bash"]