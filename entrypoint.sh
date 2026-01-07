#!/bin/bash
# FlashVSR-Pro 容器入口点脚本
# 自动激活 Conda 环境并设置所需路径

# 激活 Conda base 环境，以便使用 conda 命令
source /opt/conda/etc/profile.d/conda.sh

# 激活项目所需的 flashvsr 环境
conda activate flashvsr

# 设置 PyTorch C++ 库的运行时路径（防止 libc10.so 等库找不到的问题）
export LD_LIBRARY_PATH="/opt/conda/envs/flashvsr/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"

# 可选：进入项目目录
cd /workspace/FlashVSR

# 打印确认信息（调试时可保留，正式使用可注释掉）
echo "[INFO] Conda environment 'flashvsr' activated."
echo "[INFO] Python path: $(which python)"

# 执行 Docker 启动时传入的命令
# `exec` 使得最终进程替换当前 shell，能正确传递信号（如 SIGTERM）
exec "$@"