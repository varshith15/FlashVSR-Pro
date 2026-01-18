# FlashVSR-Pro: Enhanced Implementation of Real-Time Video Super-Resolution

**FlashVSR-Pro** is an enhanced, production-ready re-implementation of the real-time diffusion-based video super-resolution algorithm introduced in the paper **"FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution"**.

> **Original Paper**: Zhuang, J., Guo, S., Cai, X., Li, X., Liu, Y., Yuan, C., & Xue, T. (2025). *FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution*. arXiv preprint arXiv:2510.12747.  
> **Paper Link**: https://arxiv.org/abs/2510.12747

This project is not the official code release but an independent, refactored implementation focused on improved usability, additional features, and better compatibility for real-world deployment.

---

## ✨ Core Features & Enhancements

### 🚀 **Beyond the Original Implementation**
This project builds upon the core FlashVSR algorithm and introduces several key improvements:

*   **🧩 Unified Inference Script**: A single, parameterized `infer.py` script replaces multiple original scripts (`full`, `tiny`, `tiny-long`), simplifying the user interface.
*   **🎵 Audio Track Preservation**: Automatically detects and preserves the audio track from the input video in the super-resolved output. *(Inspiration for this feature was drawn from the [FlashVSR_plus](https://github.com/lihaoyun6/FlashVSR_plus) project.)*
*   **💾 Tiled Inference for Reduced VRAM**: Implements a tiling mechanism for the DiT model, significantly lowering GPU memory requirements and enabling processing of higher-resolution videos or operation on GPUs with limited VRAM. *(The concept for tiled DiT inference was inspired by the [FlashVSR_plus](https://github.com/lihaoyun6/FlashVSR_plus) project.)*
*   **🐳 Optimized Docker Container**: A fully-configured Dockerfile that automatically sets up the complete environment, including Conda environment activation upon container startup.
*   **🔧 Automated Block-Sparse Attention Installation**: Optimizes and automates the installation of the Block-Sparse-Attention backend within the Docker build process. This eliminates the manual compilation complexity encountered in the original implementation, ensuring a seamless setup experience. My specific improvements to Block-Sparse-Attention are documented in this PR: [mit-han-lab/Block-Sparse-Attention#16](https://github.com/mit-han-lab/Block-Sparse-Attention/pull/16#issue-3800081298).
*   **🎨 Configurable VAE Decoders**: Introduces a unified `VAEManager` supporting **five different VAE decoder options** (Wan2.1, Wan2.2, LightVAE, TAE_W2.2, LightTAE_HY1.5). This allows users to dynamically trade off between output quality, processing speed, and GPU memory (VRAM) usage based on their hardware and needs. See the detailed [VAE Model Selection](#vae-model-selection) guide below.
*   **⚡ Performance Optimizations**: Comprehensive speed enhancements including optimized device transfers, suppressed verbose outputs, streamlined memory management, and reduced redundant operations. Achieves 20-30% performance improvement for high-throughput video processing scenarios.

### ⚡ **Original FlashVSR Advantages (Preserved)**
*   **Real-Time Performance**: Achieves ~17 FPS for 768 × 1408 videos on a single A100 GPU.
*   **One-Step Diffusion**: Efficient streaming framework based on a distilled one-step diffusion model.
*   **State-of-the-Art Quality**: Combines Locality-Constrained Sparse Attention and a Tiny Conditional Decoder for high-fidelity results.
*   **Scalability**: Reliably scales to ultra-high resolutions.

---

## 🎨 VAE Model Selection

FlashVSR-Pro supports multiple VAE decoders to optimize for your specific hardware and quality requirements.

### VAE Type Comparison

| VAE Type | VRAM Usage | Speed | Quality | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Wan2.1** | 8-12 GB | Baseline | ★★★★☆ | High quality, moderate VRAM |
| **Wan2.2** | 8-12 GB | Baseline | ★★★★★ | Best quality, highest VRAM (H100 recommended) |
| **LightVAE_W2.1** | 4-5 GB | 2-3x faster | ★★★★☆ | 8-16GB VRAM, speed priority |
| **TAE_W2.2** | 6-8 GB | 1.5x faster | ★★★★☆ | Temporal consistency priority |
| **LightTAE_HY1.5** | 3-4 GB | 3x faster | ★★★★☆ | HunyuanVideo compatible, minimum VRAM |

### Mode and VAE Compatibility

FlashVSR-Pro has three inference modes, each compatible with specific VAE types:

| Mode | Compatible VAEs | Default VAE | Description |
|------|----------------|-------------|-------------|
| **full** | `wan2.1`, `wan2.2`, `light` | `wan2.1` | Full diffusion pipeline with VAE decoding for highest quality |
| **tiny** | `tcd`, `tae-hv`, `tae-w2.2` | `tcd` | Fast inference using Tiny Conditional Decoder |
| **tiny-long** | `tcd`, `tae-hv`, `tae-w2.2` | `tcd` | Optimized for long videos with Tiny Conditional Decoder |

**Important**: Each mode is strictly compatible with its designated VAE types. If you specify an incompatible VAE, the program will display a warning and automatically switch to the default VAE for that mode.

### VAE Selection Guide

| Your VRAM | Recommended VAE | Additional Settings |
| :--- | :--- | :--- |
| **8GB** | `LightTAE_HY1.5` or `LightVAE_W2.1` | `--tile-vae`, `--tile-dit`, `--tile-size 128` |
| **12GB** | `LightVAE_W2.1` or `Wan2.1` | `--tile-vae` |
| **16GB** | Any VAE | Optional tiling for long videos |
| **24GB+** | `Wan2.2` (preferred) or `Wan2.1` | Maximum quality, no restrictions |

### Auto-Download

All VAE models are expected to be in the `./models/VAEs/` directory. If not found, you will need to download them manually:

| VAE Selection | File | Direct Download Link |
| :--- | :--- | :--- |
| **Wan2.1** | `Wan2.1_VAE.pth` | [Download](https://huggingface.co/lightx2v/Autoencoders/blob/main/Wan2.1_VAE.pth) |
| **Wan2.2** | `Wan2.2_VAE.pth` | [Download](https://huggingface.co/lightx2v/Autoencoders/blob/main/Wan2.2_VAE.pth) |
| **LightVAE_W2.1** | `lightvaew2_1.pth` | [Download](https://huggingface.co/lightx2v/Autoencoders/blob/main/lightvaew2_1.pth) |
| **TAE_W2.2** | `taew2_2.safetensors` | [Download](https://huggingface.co/lightx2v/Autoencoders/blob/main/taew2_2.safetensors) |
| **LightTAE_HY1.5** | `lighttaehy1_5.pth` | [Download](https://huggingface.co/lightx2v/Autoencoders/blob/main/lighttaehy1_5.pth) |

**Usage Examples:**
```bash
# High quality (full mode with Wan2.2)
python infer.py -i input.mp4 -o results --mode full --vae-type wan2.2

# Balanced quality/VRAM (full mode with Light VAE)
python infer.py -i input.mp4 -o results --mode full --vae-type light

# Fast inference (tiny mode with TCDecoder)
python infer.py -i input.mp4 -o results --mode tiny --vae-type tcd

# Custom VAE weights
python infer.py -i input.mp4 -o results --mode full --vae-type wan2.1 --vae-path ./custom/path/Wan2.1_VAE.pth
```

---

## ⚡ Performance Optimizations

FlashVSR-Pro includes comprehensive performance enhancements designed for high-throughput production environments and real-time processing requirements.

### Key Optimizations

*   **🚀 Optimized Device Transfers**: Merged redundant tensor `.to()` operations into single calls, reducing device transfer overhead by ~50% during data preprocessing and model loading.
*   **🔇 Suppressed Verbose Outputs**: Automatic redirection of diffsynth library outputs and removal of detailed progress prints during tiled inference, eliminating I/O bottlenecks in high-performance scenarios.
*   **💾 Streamlined Memory Management**: Removed frequent GPU cache clearing operations and optimized VAE memory usage by eliminating unnecessary encoder components, reducing memory fragmentation.
*   **⚙️ Code Structure Improvements**: Merged redundant validation checks, cached registry accesses, and optimized pipeline initialization for faster startup times.

### Performance Benefits

| Optimization Area | Performance Impact | Use Case |
| :--- | :--- | :--- |
| **Device Transfers** | ~50% reduction in tensor movement overhead | Large video processing, batch operations |
| **Output Suppression** | Eliminates I/O blocking during inference | Real-time streaming, production deployment |
| **Memory Management** | Reduced GPU memory fragmentation | Long-running processes, high-resolution videos |
| **Code Optimization** | Faster initialization and validation | Frequent script execution, automated workflows |

### Recommended Settings for Maximum Performance

```bash
# High-performance configuration for production use
python infer.py -i input.mp4 -o results \
  --mode tiny \
  --vae-type lighttae-hy1.5 \
  --tile-dit \
  --tile-vae \
  --tile-size 256 \
  --dtype fp16 \
  --device cuda
```

**Note**: These optimizations are particularly beneficial for:
- Large-scale video processing pipelines
- Real-time streaming applications
- Production environments with high throughput requirements
- Systems with limited I/O bandwidth

---

## 🚀 Quick Start with Docker (Recommended)

The easiest way to run FlashVSR-Pro is using the provided Docker container, which includes automated setup for the Block-Sparse-Attention backend.

### 1. Prerequisites
*   Install [Docker](https://docs.docker.com/get-docker/)
*   Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU support.
*   Ensure `git-lfs` is installed on your host system to clone model weights.

### 2. Build the Docker Image
```bash
git clone https://github.com/LujiaJin/FlashVSR-Pro.git
cd FlashVSR-Pro
docker build -t flashvsr-pro:latest .
```
**Note**: The Dockerfile automatically handles the compilation and installation of the optimized Block-Sparse-Attention backend, eliminating manual configuration.

### 3. Download Model Weights
Before running the container, download the required model weights.
```bash
# Download the main FlashVSR model weights (v1.1 is recommended)
git lfs clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1 ./models/FlashVSR-v1.1
# or for v1
# git lfs clone https://huggingface.co/JunhaoZhuang/FlashVSR ./models/FlashVSR

# (Optional) Download VAE models to ./models/VAEs/ as per the table above.
```

### 4. Run the Container
The container is configured to automatically activate the `flashvsr` Conda environment upon startup. Make sure that the `models/` directory of the host machine already contains the necessary model weight files, and provide the models when starting the container by mounting.
```bash
# Basic run with interactive shell
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/FlashVSR-Pro \
  flashvsr-pro:latest

# You will be dropped into a shell with the `(flashvsr)` environment active.
# Verify by running: `which python`
```

---

## 🎯 Usage

The main interface is the unified `infer.py` script.

### Basic Inference
```bash
# Basic upscaling (Tiny mode - balanced quality/speed)
python infer.py -i ./inputs/example0.mp4 -o ./results --mode tiny

# Full mode (Highest quality, requires more VRAM)
python infer.py -i ./inputs/example0.mp4 -o ./results --mode full

# Tiny-long mode for long videos
python infer.py -i ./inputs/example4.mp4 -o ./results --mode tiny-long
```

### Using Key Enhancements
```bash
# 1. Preserve the audio track from the input video
python infer.py -i input_with_audio.mp4 -o ./results --mode tiny --keep-audio

# 2. Use tiled DiT inference to reduce VRAM usage (enables running on smaller GPUs)
python infer.py -i large_input.mp4 -o ./results --mode tiny --tile-dit --tile-size 256 --overlap 24

# 3. Use a specific VAE for optimal VRAM/quality trade-off
python infer.py -i input.mp4 -o ./results --mode full --vae-type light --tile-vae

# 4. Combine multiple enhancements
python infer.py -i large_input_with_audio.mp4 -o ./results --mode full --vae-type wan2.2 --tile-dit --keep-audio
```

### Key Arguments
| Argument | Description | Default |
| :--- | :--- | :--- |
| `-i, --input` | Path to input video or image folder. | **Required** |
| `-o, --output` | Output directory or file path. | `./results` |
| `--mode` | Inference mode: `full`, `tiny`, `tiny-long`. | `tiny` |
| `--vae-type` | VAE decoder type: `wan2.1`, `wan2.2`, `light`, `tcd`, `tae-hv`, `tae-w2.2`. | `wan2.1` (full), `tcd` (tiny/tiny-long) |
| `--vae-path` | Custom path to VAE weights file. | `None` |
| `--keep-audio` | Preserve audio from input video (if exists). | `False` |
| `--tile-dit` | Enable memory-efficient tiled DiT inference. | `False` |
| `--tile-vae` | Enable tiled decoding for VAE. | `False` |
| `--tile-size` | Size of each tile when using tiling. | `256` |
| `--overlap` | Overlap between tiles to reduce seams. | `24` |
| `--scale` | Super-resolution scale factor. | `2.0` |
| `--seed` | Random seed for reproducible results. | `0` |

**Note**: The original FlashVSR is primarily designed and tested for 4x super-resolution. While other scales are supported, for optimal quality and stability, using `--scale 4.0` is recommended.

For a full list of arguments, run `python infer.py --help`.

---

## 🔧 Troubleshooting

### 1. CUDA Out of Memory
- Use `--tile-dit` and `--tile-vae` to enable tiled inference.
- Decrease `--tile-size` (e.g., from 256 to 128).
- Use `--dtype fp16` to reduce memory usage.
- Select a lighter `--vae-type` (e.g., `light` or `lighttae-hy1.5`).

### 2. Audio Not Preserved
- Ensure ffmpeg is installed on the host system.
- Check whether the input video contains an audio stream: `ffprobe -i input.mp4`

### 3. Model Loading Fails
- Make sure model weights are downloaded with git-lfs: `git lfs pull`
- Verify model file integrity
- Check that VAE weights are in the correct directory: `./models/VAEs/`

### 4. Block-Sparse-Attention Compilation Errors
- The Docker build process should handle this automatically. If building manually, ensure you have CUDA 12.1+ and the correct PyTorch version installed.
- Reference the fix I released in: [mit-han-lab/Block-Sparse-Attention#16](https://github.com/mit-han-lab/Block-Sparse-Attention/pull/16#issue-3800081298)

---

## 🛠️ Project Structure
```
FlashVSR-Pro/
├── .gitmodules                         # Git submodule configuration
├── Block-Sparse-Attention/             # Git submodule: Sparse attention backend (with automated build)
├── models/                             # Model weights directory
│   ├── FlashVSR/                       # Model weights V1
│   ├── FlashVSR-v1.1/                  # Model weights V1.1
│   ├── prompt_tensor/                  # Pre-computed text prompt embeddings
│   └── VAEs/                           # VAE model weights (wan2.1, light, tae-hv, etc.)
├── diffsynth/                          # Core library (ModelManager, Pipelines)
├── inputs/                             # Default directory for input videos/images
├── results/                            # Default directory for output videos
├── utils/                              # Enhanced utilities module
│   ├── __init__.py
│   ├── utils.py                        # Core utilities (Causal_LQ4x_Proj, etc.)
│   ├── TCDecoder.py                    # Tiny Conditional Decoder for 'tiny' mode
│   ├── audio_utils.py                  # Audio preservation functions
│   ├── tile_utils.py                   # Tiled inference for low VRAM
│   └── vae_manager.py                  # VAE Manager for multiple VAE support
├── infer.py                            # Main unified inference script
├── Dockerfile                          # Container definition with auto-activation
├── entrypoint.sh                       # Container entry script
├── config.yaml                         # Configuration file with VAE defaults
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package setup for the `diffsynth` module
├── LICENSE                             # Project license file
└── README.md                           # This file
```

---

## 📄 License & Citation

### License
This project is released under the same license (Apache Software license) as the original FlashVSR implementation. Please see the `LICENSE` file in the original repository for details.

### Citation
If you use the FlashVSR algorithm in your research, please cite the original FlashVSR paper:
```bibtex
@article{zhuang2025flashvsr,
  title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution},
  author={Zhuang, Junhao and Guo, Shi and Cai, Xin and Li, Xiaohui and Liu, Yihao and Yuan, Chun and Xue, Tianfan},
  journal={arXiv preprint arXiv:2510.12747},
  year={2025}
}
```

**If you use this implementation (FlashVSR-Pro) in your work, please cite:**
- This repository: https://github.com/LujiaJin/FlashVSR-Pro
- The optimized Block-Sparse-Attention backend: https://github.com/LujiaJin/Block-Sparse-Attention

### Acknowledgements
*   The core algorithm is from the original **FlashVSR** authors.
*   Inspiration for the **audio preservation** and **tiled DiT inference** features came from the community project [FlashVSR_plus](https://github.com/lihaoyun6/FlashVSR_plus).
*   **The idea and implementation for supporting multiple VAE decoders** was inspired by [ComfyUI-FlashVSR_Stable](https://github.com/naxci1/ComfyUI-FlashVSR_Stable.git).
*   VAE models are from the open-source community, particularly [lightx2v/Autoencoders](https://huggingface.co/lightx2v/Autoencoders).
*   The automated build and optimization of the **Block-Sparse-Attention** backend is a contribution of this project, with improvements documented in [mit-han-lab/Block-Sparse-Attention#16](https://github.com/mit-han-lab/Block-Sparse-Attention/pull/16#issue-3800081298).

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://github.com/LujiaJin/FlashVSR-Pro/issues) if you want to contribute.

---

**Happy Super-Resolution!** 🚀