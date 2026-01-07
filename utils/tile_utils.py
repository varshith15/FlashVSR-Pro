"""
Tile utilities for FlashVSR
Provides tiled inference for DiT and VAE to reduce VRAM usage
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Optional, Generator
import warnings

def calculate_tile_coords(height: int, width: int, tile_size: int, overlap: int, multiple: int = 32) -> List[Tuple[int, int, int, int]]:
    """
    计算分块坐标 (x1, y1, x2, y2)
    
    Args:
        height: 原始高度
        width: 原始宽度
        tile_size: 分块大小
        overlap: 重叠区域大小
        multiple: 确保分块大小是该值的倍数（用于兼容模型要求）
    
    Returns:
        分块坐标列表，每个元素为 (x1, y1, x2, y2)
    """
    if overlap >= tile_size // 2:
        raise ValueError(f"Overlap ({overlap}) must be less than half of tile_size ({tile_size})")
    
    if tile_size < multiple:
        tile_size = ((tile_size + multiple - 1) // multiple) * multiple
        warnings.warn(f"Tile size increased to {tile_size} to be multiple of {multiple}")
    
    coords = []
    stride = tile_size - overlap
    
    # 计算行列数
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    
    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride
            x1 = c * stride
            
            # 初始结束坐标
            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)
            
            # 确保分块大小一致（调整结束坐标）
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
            
            # 调整到multiple的倍数
            tile_h = y2 - y1
            tile_w = x2 - x1
            
            if tile_h % multiple != 0:
                y2 = y1 + (tile_h // multiple) * multiple
                if y2 <= y1:  # 确保至少有一个倍数大小
                    y2 = y1 + multiple
            
            if tile_w % multiple != 0:
                x2 = x1 + (tile_w // multiple) * multiple
                if x2 <= x1:  # 确保至少有一个倍数大小
                    x2 = x1 + multiple
            
            # 确保不超过边界
            y2 = min(y2, height)
            x2 = min(x2, width)
            
            # 最终调整，确保至少有一个像素
            if y2 <= y1 or x2 <= x1:
                continue
                
            coords.append((x1, y1, x2, y2))
    
    # 如果没有生成任何分块，返回整个图像作为一个分块
    if not coords:
        coords = [(0, 0, width, height)]
    
    return coords

def create_feather_mask(tile_h: int, tile_w: int, overlap: int, device='cpu') -> torch.Tensor:
    """
    创建羽化遮罩（tensor版本）
    
    Args:
        tile_h: 分块高度
        tile_w: 分块宽度
        overlap: 重叠区域大小
        device: 设备
        
    Returns:
        羽化遮罩张量，形状为 (1, 1, tile_h, tile_w)
    """
    mask = torch.ones(1, 1, tile_h, tile_w, device=device, dtype=torch.float32)
    
    if overlap > 0:
        # 创建渐变
        ramp = torch.linspace(0, 1, overlap, device=device)
        
        # 左边缘
        mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
        # 右边缘
        mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
        # 上边缘
        mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
        # 下边缘
        mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    
    return mask

def create_feather_mask_numpy(tile_h: int, tile_w: int, overlap: int) -> np.ndarray:
    """
    创建羽化遮罩（numpy版本）
    
    Args:
        tile_h: 分块高度
        tile_w: 分块宽度
        overlap: 重叠区域大小
        
    Returns:
        羽化遮罩数组，形状为 (tile_h, tile_w, 1)
    """
    mask = np.ones((tile_h, tile_w, 1), dtype=np.float32)
    
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap, dtype=np.float32)
        
        # 左边缘
        mask[:, :overlap, :] *= ramp[np.newaxis, :, np.newaxis]
        # 右边缘
        mask[:, -overlap:, :] *= np.flip(ramp)[np.newaxis, :, np.newaxis]
        # 上边缘
        mask[:overlap, :, :] *= ramp[:, np.newaxis, np.newaxis]
        # 下边缘
        mask[-overlap:, :, :] *= np.flip(ramp)[:, np.newaxis, np.newaxis]
    
    return mask

def split_video_into_tiles(
    video: torch.Tensor, 
    tile_size: int, 
    overlap: int,
    multiple: int = 32
) -> Generator[Tuple[torch.Tensor, Tuple[int, int, int, int]], None, None]:
    """
    将视频张量分割成空间tiles
    
    Args:
        video: 输入视频张量，形状为 (B, C, T, H, W)
        tile_size: 分块大小
        overlap: 重叠区域大小
        multiple: 确保分块大小是该值的倍数
        
    Yields:
        (tile, (x1, y1, x2, y2)): 分块张量和其坐标
    """
    if video.dim() != 5:
        raise ValueError(f"Expected video tensor with 5 dimensions (B, C, T, H, W), got {video.dim()}")
    
    B, C, T, H, W = video.shape
    
    # 计算分块坐标
    coords = calculate_tile_coords(H, W, tile_size, overlap, multiple)
    
    for x1, y1, x2, y2 in coords:
        # 提取分块
        tile = video[:, :, :, y1:y2, x1:x2]
        yield tile, (x1, y1, x2, y2)

def stitch_video_tiles_back(
    tiles: List[torch.Tensor],
    coords: List[Tuple[int, int, int, int]],
    original_shape: Tuple[int, int],  # (H, W)
    overlap: int,
    scale: int = 1
) -> torch.Tensor:
    """
    将视频tiles拼接回原图，使用羽化融合
    
    Args:
        tiles: 分块张量列表
        coords: 分块坐标列表
        original_shape: 原始空间形状 (H, W)
        overlap: 重叠区域大小
        scale: 缩放因子（输出相对于输入的缩放）
        
    Returns:
        拼接后的视频张量，形状为 (B, C, T, H*scale, W*scale)
    """
    if not tiles or not coords:
        raise ValueError("No tiles or coordinates provided")
    
    if len(tiles) != len(coords):
        raise ValueError(f"Number of tiles ({len(tiles)}) doesn't match number of coordinates ({len(coords)})")
    
    H, W = original_shape
    H_scaled, W_scaled = H * scale, W * scale
    
    # 获取第一个tile的形状信息
    B, C, T, tile_h, tile_w = tiles[0].shape
    
    # 创建画布
    canvas = torch.zeros((B, C, T, H_scaled, W_scaled), dtype=tiles[0].dtype, device=tiles[0].device)
    weight_canvas = torch.zeros((B, 1, T, H_scaled, W_scaled), dtype=tiles[0].dtype, device=tiles[0].device)
    
    for tile, (x1, y1, x2, y2) in zip(tiles, coords):
        # 检查tile形状
        if tile.shape[2:] != (tile_h, tile_w):
            warnings.warn(f"Tile shape {tile.shape[2:]} doesn't match expected ({tile_h}, {tile_w}), resizing")
            tile = F.interpolate(tile, size=(tile_h, tile_w), mode='bilinear', align_corners=False)
        
        # 创建羽化遮罩（扩展到时间维度）
        mask = create_feather_mask(tile_h, tile_w, overlap * scale, device=tile.device)
        mask_expanded = mask.unsqueeze(2).expand(-1, -1, T, -1, -1)  # (1, 1, T, H, W)
        
        # 计算缩放后的坐标
        out_x1, out_y1 = x1 * scale, y1 * scale
        out_x2, out_y2 = out_x1 + tile_w, out_y1 + tile_h
        
        # 确保不超过画布边界
        out_y2 = min(out_y2, H_scaled)
        out_x2 = min(out_x2, W_scaled)
        
        # 调整tile大小以适应画布（如果需要）
        if out_y2 - out_y1 != tile_h or out_x2 - out_x1 != tile_w:
            tile = F.interpolate(
                tile, 
                size=(out_y2 - out_y1, out_x2 - out_x1), 
                mode='bilinear', 
                align_corners=False
            )
            mask_expanded = F.interpolate(
                mask_expanded, 
                size=(out_y2 - out_y1, out_x2 - out_x1), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 累加到画布
        canvas[:, :, :, out_y1:out_y2, out_x1:out_x2] += tile * mask_expanded
        weight_canvas[:, :, :, out_y1:out_y2, out_x1:out_x2] += mask_expanded
    
    # 归一化（避免除零）
    weight_canvas[weight_canvas == 0] = 1.0
    result = canvas / weight_canvas
    
    return result

def apply_tiled_inference_simple(
    pipeline,  # FlashVSR pipeline
    LQ_video: torch.Tensor,
    tile_size: int = 256,
    overlap: int = 24,
    **pipeline_kwargs
) -> torch.Tensor:
    """
    简化的分块推理函数，专门为FlashVSR设计
    
    Args:
        pipeline: FlashVSR pipeline实例
        LQ_video: 低质量视频张量，形状为 (1, C, T, H, W)
        tile_size: 分块大小
        overlap: 重叠区域大小
        **pipeline_kwargs: 传递给pipeline的其他参数
        
    Returns:
        超分后的视频张量
    """
    if tile_size <= 0:
        # 不使用分块
        return pipeline(**pipeline_kwargs)
    
    # 检查输入形状
    if LQ_video.dim() != 5 or LQ_video.shape[0] != 1:
        raise ValueError(f"Expected LQ_video with shape (1, C, T, H, W), got {LQ_video.shape}")
    
    B, C, T, H, W = LQ_video.shape
    
    # 计算分块坐标
    coords = calculate_tile_coords(H, W, tile_size, overlap, multiple=32)
    
    print(f"[Tiled Inference] Splitting {H}x{W} video into {len(coords)} tiles")
    
    # 存储所有分块的结果
    output_tiles = []
    
    for idx, (x1, y1, x2, y2) in enumerate(coords):
        print(f"[Tile {idx+1}/{len(coords)}] Processing tile ({x1},{y1})-({x2},{y2}), size: {y2-y1}x{x2-x1}")
        
        # 提取分块
        tile = LQ_video[:, :, :, y1:y2, x1:x2]
        
        # 更新pipeline参数
        tile_kwargs = pipeline_kwargs.copy()
        tile_kwargs['LQ_video'] = tile
        tile_kwargs['height'] = y2 - y1
        tile_kwargs['width'] = x2 - x1
        
        # 运行推理
        tile_output = pipeline(**tile_kwargs)
        output_tiles.append(tile_output)
        
        # 清理显存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 拼接所有分块
    print(f"[Tiled Inference] Stitching {len(output_tiles)} tiles back together...")
    final_output = stitch_video_tiles_back(
        output_tiles, coords, (H, W), overlap, scale=1
    )
    
    return final_output

# 为VAE解码设计的简单分块函数
def vae_decode_tiled(vae_model, latents: torch.Tensor, tile_size: int = 512, overlap: int = 32):
    """
    对VAE解码进行分块处理
    
    Args:
        vae_model: VAE模型
        latents: 潜在表示张量，形状为 (B, C, T, H, W) 或 (B, C, H, W)
        tile_size: 分块大小
        overlap: 重叠区域大小
        
    Returns:
        解码后的图像/视频张量
    """
    if latents.dim() == 4:
        # 图像情况：添加时间维度
        latents = latents.unsqueeze(2)  # (B, C, 1, H, W)
        is_image = True
    elif latents.dim() == 5:
        is_image = False
    else:
        raise ValueError(f"Expected latents with 4 or 5 dimensions, got {latents.dim()}")
    
    B, C, T, H, W = latents.shape
    
    # 如果尺寸小于tile_size，直接解码
    if H <= tile_size and W <= tile_size:
        result = vae_model.decode(latents)
        return result.squeeze(2) if is_image else result
    
    # 分块解码
    coords = calculate_tile_coords(H, W, tile_size, overlap, multiple=32)
    
    # 存储分块结果
    output_tiles = []
    
    for idx, (x1, y1, x2, y2) in enumerate(coords):
        # 提取分块
        tile = latents[:, :, :, y1:y2, x1:x2]
        
        # 解码
        tile_decoded = vae_model.decode(tile)
        output_tiles.append(tile_decoded)
    
    # 拼接
    final_output = stitch_video_tiles_back(
        output_tiles, coords, (H, W), overlap, scale=1
    )
    
    return final_output.squeeze(2) if is_image else final_output