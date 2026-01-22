#!/usr/bin/env python3
import os
import sys
import time
import argparse
import torch
import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr
import io
from einops import rearrange

# Add project path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from diffsynth import ModelManager, FlashVSRTinyPipeline
from flashvsr_utils.core.utils import Causal_LQ4x_Proj
from flashvsr_utils.vae import vae_system

def parse_args():
    parser = argparse.ArgumentParser(description="FlashVSR-Pro Tiny Inference (Stream Mode)")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input video path")
    parser.add_argument("-o", "--output", type=str, default="./results/output.mp4", help="Output path")
    parser.add_argument("--scale", type=float, default=2.0, help="Upscale factor")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()

def tensor2video(frames: torch.Tensor):
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames]
    return frames

def save_video(frames, save_path, fps=30, quality=6):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    w = imageio.get_writer(save_path, fps=fps, quality=quality, format='FFMPEG')
    for f in tqdm(frames, desc="Saving video"):
        w.append_data(np.array(f))
    w.close()

def pil_to_tensor_neg1_1(img: Image.Image, dtype=torch.bfloat16, device='cuda'):
    t = torch.from_numpy(np.asarray(img, np.uint8).copy()).to(device=device, dtype=dtype)
    t = t.permute(2,0,1) / 255.0 * 2.0 - 1.0
    return t

def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 2.0, multiple: int = 128):
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))
    tW = (sW // multiple) * multiple
    tH = (sH // multiple) * multiple
    return sW, sH, tW, tH

def upscale_then_center_crop(img: Image.Image, scale: float, tW: int, tH: int) -> Image.Image:
    w0, h0 = img.size
    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))
    up = img.resize((sW, sH), Image.BICUBIC)
    l, t = (sW - tW) // 2, (sH - tH) // 2
    return up.crop((l, t, l + tW, t + tH))

def prepare_input_tensor(path: str, scale: float = 2, dtype=torch.bfloat16, device='cuda'):
    frames = []
    fps = 30
    
    rdr = imageio.get_reader(path)
    meta = rdr.get_meta_data()
    fps = meta.get('fps', 30)

    
    img0 = Image.fromarray(rdr.get_data(0))
    w0, h0 = img0.size
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale)
    
    for i, frame in enumerate(tqdm(rdr, desc="Loading video")):
        img = Image.fromarray(frame).convert('RGB')
        img_out = upscale_then_center_crop(img, scale=scale, tW=tW, tH=tH)
        frames.append(pil_to_tensor_neg1_1(img_out, dtype, device))
    rdr.close()
        
    vid = torch.stack(frames, 0).permute(1,0,2,3).unsqueeze(0) # (1, C, T, H, W)
    return vid, tH, tW, vid.shape[2], fps

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    print(f"Device: {device}")
    
    print(f"Loading input: {args.input}")
    LQ, tH, tW, F, fps = prepare_input_tensor(args.input, scale=args.scale, dtype=dtype, device=device)
    print(f"Input loaded. Frames: {F}, Target Size: {tW}x{tH}, FPS: {fps}")

    model_dir = os.getenv("FLASHVSR_MODEL_PATH", "./models/FlashVSR-v1.1")
    
    vae_manager = vae_system.VAESystem(device=device, dtype=dtype)
    vae_model = vae_manager.load_vae(vae_type="tcd", model_dir=model_dir)
    
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    dit_path = f"{model_dir}/diffusion_pytorch_model_streaming_dmd.safetensors"
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        mm.load_models([dit_path])
        pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
    
    pipe.TCDecoder = vae_model
    
    lq_proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
    lq_path = f"{model_dir}/LQ_proj_in.ckpt"
    if os.path.exists(lq_path):
        lq_proj.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in = lq_proj.to(device, dtype=dtype)
    
    pipe.to(device)
    pipe.init_cross_kv()
    
    output_frames = []
    
    dummy_frames = torch.randn(1, 3, 25, tH, tW, device=device, dtype=dtype).clamp(-1, 1)
    _ = pipe.stream(dummy_frames, height=tH, width=tW, seed=args.seed)
    remainder = F % 8
    if remainder != 0:
        pad_len = 8 - remainder
        padding = LQ[:, :, -1:, :, :].repeat(1, 1, pad_len, 1, 1)
        LQ = torch.cat([LQ, padding], dim=2)
        F += pad_len
        print(f"Padded video to {F} frames (multiple of 8)")
    
    latencies = []
    fps_list = []
    num_chunks = F // 8
    
    for i in range(num_chunks):
        start_idx = i * 8
        end_idx = start_idx + 8
        chunk = LQ[:, :, start_idx:end_idx, :, :]
        
        t0 = time.time()
        out_chunk = pipe.stream(chunk, height=tH, width=tW, seed=args.seed)
        t1 = time.time()
        
        latency = t1 - t0
        fps_chunk = 8 / latency
        latencies.append(latency)
        fps_list.append(fps_chunk)
        
        print(f"Chunk {i+1}/{num_chunks} (Frames {start_idx}-{end_idx}): FPS={fps_chunk:.2f}")
        
        output_frames.append(out_chunk.cpu())
        
    avg_fps = sum(fps_list) / len(fps_list)
    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage FPS: {avg_fps:.2f}")
    print(f"Average Latency (per 8 frames): {avg_latency:.4f}s")
    
    # 4. Save
    print("Saving output video...")
    full_tensor = torch.cat(output_frames, dim=2)
    
    video_frames = tensor2video(full_tensor[0])
    save_video(video_frames, args.output, fps=fps)
    print(f"Done! Saved to {args.output}")

if __name__ == "__main__":
    main()
