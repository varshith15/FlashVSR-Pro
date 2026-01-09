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
	Compute tile coordinates (x1, y1, x2, y2)
	
	Args:
		height: original height
		width: original width
		tile_size: tile size
		overlap: overlap size
		multiple: ensure tile dimensions are multiples of this value (model compatibility)
	
	Returns:
		List of tile coordinates, each as (x1, y1, x2, y2)
	"""
	if overlap >= tile_size // 2:
		raise ValueError(f"Overlap ({overlap}) must be less than half of tile_size ({tile_size})")
	
	if tile_size < multiple:
		tile_size = ((tile_size + multiple - 1) // multiple) * multiple
		warnings.warn(f"Tile size increased to {tile_size} to be multiple of {multiple}")
	
	coords = []
	stride = tile_size - overlap
	
	# Compute number of rows and columns
	num_rows = math.ceil((height - overlap) / stride)
	num_cols = math.ceil((width - overlap) / stride)
	
	for r in range(num_rows):
		for c in range(num_cols):
			y1 = r * stride
			x1 = c * stride
			
			# initial end coordinates
			y2 = min(y1 + tile_size, height)
			x2 = min(x1 + tile_size, width)
			
			# Ensure consistent tile size (adjust end coordinates)
			if y2 - y1 < tile_size:
				y1 = max(0, y2 - tile_size)
			if x2 - x1 < tile_size:
				x1 = max(0, x2 - tile_size)
			
			# Adjust to be multiple of 'multiple'
			tile_h = y2 - y1
			tile_w = x2 - x1
			
			if tile_h % multiple != 0:
				y2 = y1 + (tile_h // multiple) * multiple
				if y2 <= y1:  # Ensure at least one multiple-sized block
					y2 = y1 + multiple
			
			if tile_w % multiple != 0:
				x2 = x1 + (tile_w // multiple) * multiple
				if x2 <= x1:  # Ensure at least one multiple-sized block
					x2 = x1 + multiple
			
			# Ensure not exceeding boundaries
			y2 = min(y2, height)
			x2 = min(x2, width)
			
			# Final check, ensure at least one pixel
			if y2 <= y1 or x2 <= x1:
				continue
				
			coords.append((x1, y1, x2, y2))
	
	# If no tiles generated, return whole image as one tile
	if not coords:
		coords = [(0, 0, width, height)]
	
	return coords

def create_feather_mask(tile_h: int, tile_w: int, overlap: int, device='cpu') -> torch.Tensor:
	"""
	Create a feathering mask (tensor version)
	
	Args:
		tile_h: tile height
		tile_w: tile width
		overlap: overlap size
		device: device
		
    Returns:
        Feathering mask tensor with shape (1, 1, tile_h, tile_w)
	"""
	mask = torch.ones(1, 1, tile_h, tile_w, device=device, dtype=torch.float32)
	
	if overlap > 0:
		# create ramp
		ramp = torch.linspace(0, 1, overlap, device=device)
		
		# left edge
		mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
		# right edge
		mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
		# top edge
		mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
		# bottom edge
		mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
	
	return mask

def create_feather_mask_numpy(tile_h: int, tile_w: int, overlap: int) -> np.ndarray:
	"""
	Create a feathering mask (numpy version)
	
	Args:
		tile_h: tile height
		tile_w: tile width
		overlap: overlap size
		
    Returns:
        Feathering mask array with shape (tile_h, tile_w, 1)
	"""
	mask = np.ones((tile_h, tile_w, 1), dtype=np.float32)
	
	if overlap > 0:
		ramp = np.linspace(0, 1, overlap, dtype=np.float32)
		
		# left edge
		mask[:, :overlap, :] *= ramp[np.newaxis, :, np.newaxis]
		# right edge
		mask[:, -overlap:, :] *= np.flip(ramp)[np.newaxis, :, np.newaxis]
		# top edge
		mask[:overlap, :, :] *= ramp[:, np.newaxis, np.newaxis]
		# bottom edge
		mask[-overlap:, :, :] *= np.flip(ramp)[:, np.newaxis, np.newaxis]
	
	return mask

def split_video_into_tiles(
	video: torch.Tensor, 
	tile_size: int, 
	overlap: int,
	multiple: int = 32
) -> Generator[Tuple[torch.Tensor, Tuple[int, int, int, int]], None, None]:
	"""
	Split a video tensor into spatial tiles
	
	Args:
		video: input video tensor of shape (B, C, T, H, W)
		tile_size: tile size
		overlap: overlap size
		multiple: ensure tile sizes are multiples of this value
		
    Yields:
		(tile, (x1, y1, x2, y2)): tile tensor and its coordinates
	"""
	if video.dim() != 5:
		raise ValueError(f"Expected video tensor with 5 dimensions (B, C, T, H, W), got {video.dim()}")
	
	B, C, T, H, W = video.shape
	
	# compute tile coordinates
	coords = calculate_tile_coords(H, W, tile_size, overlap, multiple)
	
	for x1, y1, x2, y2 in coords:
		# extract tile
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
	Stitch video tiles back into the full frame using feathered blending
	
	Args:
		tiles: list of tile tensors
		coords: list of tile coordinates
		original_shape: original spatial shape (H, W)
		overlap: overlap size
		scale: scale factor (output relative to input)
		
    Returns:
        Stitched video tensor of shape (B, C, T, H*scale, W*scale)
	"""
	if not tiles or not coords:
		raise ValueError("No tiles or coordinates provided")
	
	if len(tiles) != len(coords):
		raise ValueError(f"Number of tiles ({len(tiles)}) doesn't match number of coordinates ({len(coords)})")
	
	H, W = original_shape
	H_scaled, W_scaled = H * scale, W * scale
	
	# get shape info from first tile
	B, C, T, tile_h, tile_w = tiles[0].shape
	
	# create canvas
	canvas = torch.zeros((B, C, T, H_scaled, W_scaled), dtype=tiles[0].dtype, device=tiles[0].device)
	weight_canvas = torch.zeros((B, 1, T, H_scaled, W_scaled), dtype=tiles[0].dtype, device=tiles[0].device)
	
	for tile, (x1, y1, x2, y2) in zip(tiles, coords):
		# check tile shape
		if tile.shape[2:] != (tile_h, tile_w):
			warnings.warn(f"Tile shape {tile.shape[2:]} doesn't match expected ({tile_h}, {tile_w}), resizing")
			tile = F.interpolate(tile, size=(tile_h, tile_w), mode='bilinear', align_corners=False)
		
		# create feather mask (expanded to time dimension)
		mask = create_feather_mask(tile_h, tile_w, overlap * scale, device=tile.device)
		mask_expanded = mask.unsqueeze(2).expand(-1, -1, T, -1, -1)  # (1, 1, T, H, W)
		
		# compute scaled coordinates
		out_x1, out_y1 = x1 * scale, y1 * scale
		out_x2, out_y2 = out_x1 + tile_w, out_y1 + tile_h
		
		# ensure not exceeding canvas bounds
		out_y2 = min(out_y2, H_scaled)
		out_x2 = min(out_x2, W_scaled)
		
		# resize tile to fit canvas if needed
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
		
		# accumulate onto canvas
		canvas[:, :, :, out_y1:out_y2, out_x1:out_x2] += tile * mask_expanded
		weight_canvas[:, :, :, out_y1:out_y2, out_x1:out_x2] += mask_expanded
	
	# add small epsilon before normalization to avoid division by zero
	weight_canvas[weight_canvas == 0] = 1.0
	epsilon = 1e-8  # add small epsilon
	result = canvas / (weight_canvas + epsilon)  # add epsilon to denom
	
	return result

def apply_tiled_inference_simple(
	pipeline,  # FlashVSR pipeline
	LQ_video: torch.Tensor,
	tile_size: int = 256,
	overlap: int = 24,
	**pipeline_kwargs
) -> torch.Tensor:
	"""
	Simplified tiled inference function designed for FlashVSR
	
	Args:
		pipeline: FlashVSR pipeline instance
		LQ_video: low-quality video tensor with shape (1, C, T, H, W)
		tile_size: tile size
		overlap: overlap size
		**pipeline_kwargs: additional args passed to pipeline
		
    Returns:
        Super-resolved video tensor
	"""
	if tile_size <= 0:
		# don't use tiling
		return pipeline(**pipeline_kwargs)
	
	# handle 4D input by adding a batch dimension
	if LQ_video.dim() == 4:
		LQ_video = LQ_video.unsqueeze(0)  # add a leading dimension to become (1, C, H, W)
	
	# check input shape
	if LQ_video.dim() != 5 or LQ_video.shape[0] != 1:
		raise ValueError(f"Expected LQ_video with shape (1, C, T, H, W), got {LQ_video.shape}")
	
	B, C, T, H, W = LQ_video.shape
	
	# compute tile coordinates
	coords = calculate_tile_coords(H, W, tile_size, overlap, multiple=32)
	
	print(f"[Tiled Inference] Splitting {H}x{W} video into {len(coords)} tiles")
	
	# store outputs of tiles
	output_tiles = []
	
	for idx, (x1, y1, x2, y2) in enumerate(coords):
		print(f"[Tile {idx+1}/{len(coords)}] Processing tile ({x1},{y1})-({x2},{y2}), size: {y2-y1}x{x2-x1}")
		
		# extract tile
		tile = LQ_video[:, :, :, y1:y2, x1:x2]
		
		# update pipeline parameters
		tile_kwargs = pipeline_kwargs.copy()
		tile_kwargs['LQ_video'] = tile
		tile_kwargs['height'] = y2 - y1
		tile_kwargs['width'] = x2 - x1
		
		# run inference
		tile_output = pipeline(**tile_kwargs)
		output_tiles.append(tile_output)
		
		# clear CUDA cache
		torch.cuda.empty_cache() if torch.cuda.is_available() else None
	
	# stitch all tiles
	final_output = stitch_video_tiles_back(
		output_tiles, coords, (H, W), overlap, scale=1
	)
	
	return final_output

# Simple tiling function designed for VAE decoding
def vae_decode_tiled(vae_model, latents: torch.Tensor, tile_size: int = 512, overlap: int = 32):
	"""
	Perform tiled decoding for the VAE
	
	Args:
		vae_model: VAE model
		latents: latent tensor of shape (B, C, T, H, W) or (B, C, H, W)
		tile_size: tile size
		overlap: overlap size
		
    Returns:
        Decoded image/video tensor
	"""
	if latents.dim() == 4:
		# image case: add time dimension
		latents = latents.unsqueeze(2)  # (B, C, 1, H, W)
		is_image = True
	elif latents.dim() == 5:
		is_image = False
	else:
		raise ValueError(f"Expected latents with 4 or 5 dimensions, got {latents.dim()}")
	
	B, C, T, H, W = latents.shape
	
	# if size <= tile_size, decode directly
	if H <= tile_size and W <= tile_size:
		result = vae_model.decode(latents)
		return result.squeeze(2) if is_image else result
	
	# tiled decode
	coords = calculate_tile_coords(H, W, tile_size, overlap, multiple=32)
	
	# store tile results
	output_tiles = []
	
	for idx, (x1, y1, x2, y2) in enumerate(coords):
		# extract tile
		tile = latents[:, :, :, y1:y2, x1:x2]
		
		# decode
		tile_decoded = vae_model.decode(tile)
		output_tiles.append(tile_decoded)
	
	# stitch
	final_output = stitch_video_tiles_back(
		output_tiles, coords, (H, W), overlap, scale=1
	)
	
	return final_output.squeeze(2) if is_image else final_output