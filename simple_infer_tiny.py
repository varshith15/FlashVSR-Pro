#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import time
import argparse
import warnings
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import io

import numpy as np
from PIL import Image
import imageio
from tqdm import tqdm
import torch
from einops import rearrange

# Add project path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from diffsynth import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
from utils.utils import Causal_LQ4x_Proj
from utils.TCDecoder import build_tcdecoder
from utils.vae_manager import VAEManager

# Optional audio utilities
try:
    from utils.audio_utils import copy_video_with_audio, has_audio_stream
    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
    warnings.warn(f"Audio utilities not available: {e}")
    
    # Provide simple fallback functions
    def has_audio_stream(path):
        return False
    
    def copy_video_with_audio(original_video_path, processed_video_path, output_path):
        import shutil
        shutil.copy2(processed_video_path, output_path)
        return True

# Tile utilities
try:
    from utils.tile_utils import calculate_tile_coords, apply_tiled_inference_simple
    TILE_AVAILABLE = True
except ImportError as e:
    TILE_AVAILABLE = False
    warnings.warn(f"Tile utilities not available: {e}")
    
    # Provide simple fallback functions
    def calculate_tile_coords(height, width, tile_size, overlap):
        return [(0, 0, width, height)]
    
    def apply_tiled_inference_simple(pipeline, LQ_video, tile_size=256, overlap=24, **pipeline_kwargs):
        # No tiling, call pipeline directly
        return pipeline(**pipeline_kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description="FlashVSR-Pro Inference Script")
    
    # Basic parameters
    parser.add_argument("-i", "--input", type=str, required=True,
                       help="Path to input video file or folder of images")
    parser.add_argument("-o", "--output", type=str, default="./results",
                       help="Output directory or file path")
    parser.add_argument("--mode", type=str, default="tiny", 
                       choices=["full", "tiny", "tiny-long"],
                       help="Inference mode: full (with Wan VAE), tiny (with TCDecoder), tiny-long (for long videos)")
    parser.add_argument("--version", type=str, default="1.1",
                       choices=["1", "1.1"],
                       help="Model version")
    
    # VAE selection parameters
    parser.add_argument("--vae-type", type=str, default=None,
                       choices=["wan2.1", "wan2.2", "light", "tcd", "tae-hv", "tae-w2.2"],
                       help="Select VAE decoder type (wan2.1/wan2.2/light for full mode, tcd/tae-hv/tae-w2.2 for tiny modes)")
    parser.add_argument("--vae-path", type=str, default=None,
                       help="Custom path to VAE weights file (overrides default)")
    
    # Tiling parameters
    parser.add_argument("--tile-dit", action="store_true",
                       help="Enable tiled inference for DiT (reduces VRAM usage)")
    parser.add_argument("--tile-vae", action="store_true",
                       help="Enable tiled decoding for VAE (only for full mode)")
    parser.add_argument("--tile-size", type=int, default=256,
                       help="Tile size for tiled inference")
    parser.add_argument("--overlap", type=int, default=24,
                       help="Overlap size between tiles")
    
    # Audio parameters
    parser.add_argument("--keep-audio", action="store_true",
                       help="Keep audio from input video (if exists)")
    
    # Inference parameters
    parser.add_argument("--scale", type=float, default=2.0,
                       help="Upscale factor")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--sparse-ratio", type=float, default=2.0,
                       help="Sparse attention ratio (1.5=faster, 2.0=stable)")
    parser.add_argument("--kv-ratio", type=float, default=3.0,
                       help="KV cache ratio")
    parser.add_argument("--local-range", type=int, default=11,
                       help="Local attention range (9=sharper, 11=stable)")
    parser.add_argument("--color-fix", action="store_true",
                       help="Apply color correction")
    parser.add_argument("--fps", type=int, default=30,
                       help="Output FPS (for image sequences)")
    parser.add_argument("--quality", type=int, default=6,
                       help="Output video quality (0-10)")
    
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bf16",
                       choices=["fp32", "fp16", "bf16"],
                       help="Data type")
    
    return parser.parse_args()

def tensor2video(frames: torch.Tensor):
    """Convert tensor to list of PIL Images"""
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames

def natural_key(name: str):
    """Natural sort key for filenames"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    """List image files with natural sorting"""
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def largest_8n1_leq(n):
    """Find largest 8n+1 <= n"""
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def is_video(path):
    """Check if path is a video file"""
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv','.webm'))

def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    """Convert PIL Image to tensor in range [-1, 1]"""
    t = torch.from_numpy(np.asarray(img, np.uint8).copy()).to(device=device, dtype=dtype)
    t = t.permute(2,0,1) / 255.0 * 2.0 - 1.0  # CHW in [-1,1]
    return t

def save_video(frames, save_path, fps=30, quality=5):
    """Save frames as video"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    for f in tqdm(frames, desc=f"Saving {os.path.basename(save_path)}"):
        w.append_data(np.array(f))
    w.close()

def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 2.0, multiple: int = 128):
    """Compute scaled dimensions and target dimensions"""
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(
            f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )

    return sW, sH, tW, tH

def upscale_then_center_crop(img: Image.Image, scale: float, tW: int, tH: int) -> Image.Image:
    """Upscale and center crop image"""
    w0, h0 = img.size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))

    if tW > sW or tH > sH:
        raise ValueError(
            f"Target crop ({tW}x{tH}) exceeds scaled size ({sW}x{sH}). "
            f"Increase scale."
        )

    up = img.resize((sW, sH), Image.BICUBIC)
    l = (sW - tW) // 2
    t = (sH - tH) // 2
    return up.crop((l, t, l + tW, t + tH))

def prepare_input_tensor(path: str, scale: float = 2, dtype=torch.bfloat16, device='cuda'):
    """Prepare input tensor from video or image sequence"""
    if os.path.isdir(path):
        # Image sequence
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")

        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        N0 = len(paths0)
        print(f"Input: {w0}x{h0}, {N0} frames")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"Scaled: {sW}x{sH} -> {tW}x{tH}")

        paths = paths0 + [paths0[-1]] * 4
        F = largest_8n1_leq(len(paths))
        if F == 0:
            raise RuntimeError(f"Not enough frames after padding. Got {len(paths)}.")
        paths = paths[:F]
        print(f"Frames: {F-4} (padded to {F})")

        frames = []
        for p in paths:
            with Image.open(p).convert('RGB') as img:
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
            frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)
        fps = 30
        return vid, tH, tW, F, fps, None

    if is_video(path):
        # Video file
        rdr = imageio.get_reader(path)
        first = Image.fromarray(rdr.get_data(0)).convert('RGB')
        w0, h0 = first.size

        meta = {}
        try: 
            meta = rdr.get_meta_data()
        except Exception: 
            pass
        
        fps_val = meta.get('fps', 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30

        def count_frames(r):
            try:
                nf = meta.get('nframes', None)
                if isinstance(nf, int) and nf > 0: 
                    return nf
            except Exception: 
                pass
            try: 
                return r.count_frames()
            except Exception:
                n = 0
                try:
                    while True: 
                        r.get_data(n)
                        n += 1
                except Exception:
                    return n

        total = count_frames(rdr)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")

        print(f"Input: {w0}x{h0}, {total} frames, {fps} FPS")

        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        print(f"Scaled: {sW}x{sH} -> {tW}x{tH}")

        idx = list(range(total)) + [total-1] * 4
        F = largest_8n1_leq(len(idx))
        if F == 0:
            rdr.close()
            raise RuntimeError(f"Not enough frames after padding. Got {len(idx)}.")
        idx = idx[:F]
        print(f"Frames: {F-4} (padded to {F})")

        frames = []
        try:
            for i in idx:
                img = Image.fromarray(rdr.get_data(i)).convert('RGB')
                img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
                frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
        finally:
            try: 
                rdr.close()
            except Exception: 
                pass

        vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0)
        return vid, tH, tW, F, fps, path

    raise ValueError(f"Unsupported input: {path}")

def init_pipeline(args):
    """Initialize pipeline based on mode and version"""
    print(f"Device: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # Determine model path
    model_dir = os.getenv("FLASHVSR-Pro_MODEL_PATH", 
                     f"./models/FlashVSR-v{args.version}" if args.version == "1.1" else "./models/FlashVSR")
    
    # Setup dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)
    
    # Initialize VAE Manager and load VAE
    vae_manager = VAEManager(device=args.device, dtype=dtype)
    vae_model = vae_manager.load_vae(
        vae_type=args.vae_type,
        custom_path=args.vae_path,
        mode=args.mode,
        tile_vae=args.tile_vae,
        tile_size=args.tile_size,
        overlap=args.overlap,
        model_dir=model_dir
    )

    # Initialize model manager and load DiT model
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    dit_path = f"{model_dir}/diffusion_pytorch_model_streaming_dmd.safetensors"
    
    # Create pipeline based on mode (suppress verbose output)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        mm.load_models([dit_path])
        
        if args.mode == "full":
            pipe = FlashVSRFullPipeline.from_model_manager(mm, device=args.device)
            pipe.vae = vae_model
        else:  # tiny or tiny-long
            if args.mode == "tiny":
                pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=args.device)
            else:  # tiny-long
                pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=args.device)
            pipe.TCDecoder = vae_model
    
    # Load and setup LQ projector efficiently
    lq_proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
    lq_path = f"{model_dir}/LQ_proj_in.ckpt"
    if os.path.exists(lq_path):
        lq_proj.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    
    # Move to device and setup pipeline (combined operations)
    pipe.denoising_model().LQ_proj_in = lq_proj.to(args.device, dtype=dtype)
    pipe.to(args.device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    
    return pipe, vae_manager

def main():
    args = parse_args()

    # Set default VAE based on mode if not specified
    if args.vae_type is None:
        if args.mode == "full":
            args.vae_type = "wan2.1"
        else:  # tiny or tiny-long
            args.vae_type = "tcd"

    # Validate VAE compatibility with mode (cache registry to avoid repeated imports)
    vae_registry = VAEManager.VAE_REGISTRY
    is_tcdecoder = vae_registry[args.vae_type]["is_tcdecoder"]
    
    if args.mode == "full" and is_tcdecoder:
        print(f"[Warning] VAE type '{args.vae_type}' is not compatible with 'full' mode.")
        print(f"[Warning] Automatically switching to default VAE 'wan2.1' for full mode.")
        args.vae_type = "wan2.1"
        is_tcdecoder = False  # Update flag
    elif args.mode in ["tiny", "tiny-long"] and not is_tcdecoder:
        print(f"[Warning] VAE type '{args.vae_type}' is not compatible with '{args.mode}' mode.")
        print(f"[Warning] Automatically switching to default VAE 'tcd' for {args.mode} mode.")
        args.vae_type = "tcd"
        is_tcdecoder = True  # Update flag

    # Combined device and memory checks
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("[WARNING] CUDA not available, falling back to CPU")
            args.device = "cpu"
        else:
            # Check GPU memory only if CUDA is available and not using tile-dit
            if not args.tile_dit:
                free_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                if free_mem < 8:  # less than 8GB
                    print(f"[WARNING] Low GPU memory ({free_mem:.1f}GB), consider using --tile-dit")
    
    # Check audio utility availability
    if args.keep_audio and not AUDIO_AVAILABLE:
        print("[Warning] Audio preservation requested but audio utilities not available.")
        print("[Warning] Install ffmpeg and ensure it's in PATH for audio support.")
        args.keep_audio = False
    
    # Check tile utility availability
    if (args.tile_dit or args.tile_vae) and not TILE_AVAILABLE:
        print("[Warning] Tiled inference requested but tile utilities not available.")
        print("[Warning] Continuing without tiling (may cause high VRAM usage).")
        args.tile_dit = False
        args.tile_vae = False
    
    # Automatically adjust too-small tile-size to avoid window attention errors
    if args.tile_dit and args.tile_size < 128:
        print(f"[WARNING] tile-size {args.tile_size} is too small and may cause window attention errors.")
        print(f"[WARNING] Setting tile-size to 128 (minimum supported value).")
        args.tile_size = 128

    # Create output directory
    if os.path.isdir(args.output):
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Prepare input
    print(f"Processing: {args.input}")
    
    # Set dtype
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    LQ, th, tw, F, fps, input_video_path = prepare_input_tensor(
        args.input, 
        scale=args.scale, 
        dtype=dtype,
        device=args.device
    )
    
    # Initialize pipeline with VAE manager
    pipe, vae_manager = init_pipeline(args)
    
    # Determine output file name
    input_name = os.path.basename(args.input.rstrip('/')).split('.')[0]
    if os.path.isdir(args.output):
        output_filename = f"FlashVSR-Pro_{args.mode}_{input_name}_seed{args.seed}.mp4"
        output_path = os.path.join(args.output, output_filename)
    else:
        output_path = args.output
    
    print(f"Output: {output_path}")
    
    # Prepare pipeline parameters
    pipeline_kwargs = {
        "prompt": "", 
        "negative_prompt": "", 
        "cfg_scale": 1.0, 
        "num_inference_steps": 1, 
        "seed": args.seed,
        "LQ_video": LQ, 
        "num_frames": F, 
        "height": th, 
        "width": tw, 
        "is_full_block": False, 
        "if_buffer": True,
        "topk_ratio": args.sparse_ratio * 768 * 1280 / (th * tw), 
        "kv_ratio": args.kv_ratio,
        "local_range": args.local_range,
        "color_fix": args.color_fix,
    }
    
    # Add VAE tiling parameters (only for full mode)
    if args.mode == "full" and args.tile_vae:
        pipeline_kwargs["tiled"] = True
    
    # Run inference (tiled or standard)
    if args.tile_dit:
        print(f"Tiled DiT: tile_size={args.tile_size}, overlap={args.overlap}")
        
        # Create a copy of pipeline_kwargs and remove LQ_video
        tile_kwargs = pipeline_kwargs.copy()
        tile_kwargs.pop('LQ_video', None)  # Remove LQ_video because it's already passed as a positional argument

        # Tiled inference
        video = apply_tiled_inference_simple(
            pipe,
            LQ,
            tile_size=args.tile_size,
            overlap=args.overlap,
            **tile_kwargs
        )
    else:
        print("Running inference...")
        video = pipe(**pipeline_kwargs)
    
    # Convert and save video
    frames = tensor2video(video)
    
    # Save temporary video file (without audio)
    temp_output = output_path.replace('.mp4', '_temp.mp4')
    save_video(frames, temp_output, fps=fps, quality=args.quality)
    print(f"Video saved: {temp_output}")
    
    # Audio handling
    if args.keep_audio and input_video_path and is_video(input_video_path):
        if has_audio_stream(input_video_path):
            print("Preserving audio...")
            # Merge audio from original video with processed video
            copy_video_with_audio(input_video_path, temp_output, output_path)
            
            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
        else:
            print("No audio found, saving silent video")
            if os.path.exists(temp_output):
                os.rename(temp_output, output_path)
    else:
        # No audio preservation requested or not a video file
        if os.path.exists(temp_output):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_output, output_path)

    print(f"Done! Output: {output_path}")

    # Cleanup
    vae_manager.clean_memory()
    del pipe
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()