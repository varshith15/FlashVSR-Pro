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

### ⚡ **Original FlashVSR Advantages (Preserved)**
*   **Real-Time Performance**: Achieves ~17 FPS for 768 × 1408 videos on a single A100 GPU.
*   **One-Step Diffusion**: Efficient streaming framework based on a distilled one-step diffusion model.
*   **State-of-the-Art Quality**: Combines Locality-Constrained Sparse Attention and a Tiny Conditional Decoder for high-fidelity results.
*   **Scalability**: Reliably scales to ultra-high resolutions.

---

## 🚀 Quick Start with Docker (Recommended)

The easiest way to run FlashVSR-Pro is using the provided Docker container.

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

### 3. Download Model Weights
Before running the container, download the required model weights using Git LFS.
```bash
# Install Git LFS if not already installed
git lfs install

# Clone the desired model version (v1.1 is recommended)
git lfs clone https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1
# or for v1
# git lfs clone https://huggingface.co/JunhaoZhuang/FlashVSR
```

### 4. Run the Container
The container is configured to automatically activate the `flashvsr` Conda environment upon startup.
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

# 3. Combine audio preservation and tiled inference
python infer.py -i large_input_with_audio.mp4 -o ./results --mode full --tile-dit --keep-audio
```

### Key Arguments
| Argument | Description | Default |
| :--- | :--- | :--- |
| `-i, --input` | Path to input video or image folder. | **Required** |
| `-o, --output` | Output directory path. | `./results` |
| `--mode` | Inference mode: `full`, `tiny`, `tiny-long`. | `tiny` |
| `--keep-audio` | Preserve audio from input video (if exists). | `False` |
| `--tile-dit` | Enable memory-efficient tiled DiT inference. | `False` |
| `--tile-size` | Size of each tile when using `--tile-dit`. | `256` |
| `--overlap` | Overlap between tiles to reduce seams. | `24` |
| `--scale` | Super-resolution scale factor. | `2.0` |
| `--seed` | Random seed for reproducible results. | `0` |

Note: The original FlashVSR is primarily designed and tested for 4x super-resolution. While other scales are supported, for optimal quality and stability, using `--scale 4.0` is recommended.

For a full list of arguments, run `python infer.py --help`.

---

## 🔧 Troubleshooting

### 1. CUDA Out of Memory
- Use `--tile-dit` to enable tiled inference.
- Decrease `--tile-size` (e.g. from 256 to 128).
- Use `--dtype fp16` to reduce memory usage.

### 2. Audio Not Preserved
- Ensure ffmpeg is installed on the host system.
- Check whether the input video contains an audio stream: `ffprobe -i input.mp4`

### 3. Model Loading Fails
- Make sure model weights are downloaded with git-lfs: `git lfs pull`
- Verify model file integrity

---

## 🛠️ Project Structure
Below is the overview of the **FlashVSR-Pro** project. It adopts a flat, user-friendly structure compared to the original nested layout.
```
FlashVSR-Pro/
├── .gitmodules                         # Git submodule configuration
├── Block-Sparse-Attention/             # Git submodule: Sparse attention backend
├── model/                              # **Model weights directory**
│   ├── FlashVSR/                       # Model weights V1 (to be downloaded by user)
│   ├── FlashVSR-v1.1/                  # Model weights V1.1 (to be downloaded by user)
│   ├── prompt_tensor/                  # Pre-computed text prompt embeddings
│   │   └── posi_prompt.pth
├── diffsynth/                          # Core library (ModelManager, Pipelines)
├── inputs/                             # **Default directory for input videos/images**
├── results/                            # **Default directory for output videos**
├── utils/                              # Enhanced utilities module
│   ├── __init__.py
│   ├── utils.py                        # Core utilities (Causal_LQ4x_Proj, etc.)
│   ├── TCDecoder.py                    # Tiny Conditional Decoder for 'tiny' mode
│   ├── audio_utils.py                  # **New**: Audio preservation functions
│   └── tile_utils.py                   # **New**: Tiled inference for low VRAM
├── infer.py                            # **Main unified inference script**
├── Dockerfile                          # Container definition with auto-activation
├── entrypoint.sh                       # Container entry script
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
If you use FlashVSR algorithm in your research, please cite the original FlashVSR paper:
```bibtex
@article{zhuang2025flashvsr,
  title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution},
  author={Zhuang, Junhao and Guo, Shi and Cai, Xin and Li, Xiaohui and Liu, Yihao and Yuan, Chun and Xue, Tianfan},
  journal={arXiv preprint arXiv:2510.12747},
  year={2025}
}
```

And if you use this implementation in your work, Please cite this repository: https://github.com/LujiaJin/FlashVSR-Pro.

### Acknowledgements
*   The core algorithm is from the original **FlashVSR** authors.
*   Inspiration for the **audio preservation** and **tiled DiT inference** features came from the community project [FlashVSR_plus](https://github.com/lihaoyun6/FlashVSR_plus).

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://github.com/LujiaJin/FlashVSR-Pro/issues) if you want to contribute.

---

**Happy Super-Resolution!** 🚀