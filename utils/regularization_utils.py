#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from simple_knn._C import distCUDA2


def albedo_smooth_loss(gaussians, k=16, weight=0.01, sample_size=2048):
    """
    Albedo 3D spatial smoothness regularization.
    Encourages neighboring Gaussians to have similar albedo values.
    
    Args:
        gaussians: GaussianModel instance
        k: number of nearest neighbors (not used in simplified version)
        weight: loss weight
        sample_size: number of points to sample for efficiency
    
    Returns:
        Weighted smoothness loss
    """
    xyz = gaussians.get_xyz
    albedo = gaussians.get_albedo
    N = xyz.shape[0]
    
    if N == 0:
        return torch.tensor(0.0, device="cuda")
    
    # Reduce sample size to avoid OOM
    actual_sample_size = min(sample_size, N, 2048)
    
    # Random sampling for efficiency
    if N > actual_sample_size:
        indices = torch.randperm(N, device=xyz.device)[:actual_sample_size]
        xyz_sample = xyz[indices]
        albedo_sample = albedo[indices]
    else:
        xyz_sample = xyz
        albedo_sample = albedo
        actual_sample_size = N
    
    # Compute average nearest neighbor distance for radius estimation
    try:
        dists_sq = distCUDA2(xyz_sample)  # (actual_sample_size,) squared distances to nearest neighbor
        dists = torch.sqrt(torch.clamp(dists_sq, min=1e-8))
        radius = dists.mean() * 2.0  # Use 2x average nearest neighbor distance
    except:
        # Fallback if KNN fails
        radius = 0.1
    
    # Memory-efficient neighborhood computation using chunking
    chunk_size = 512  # Process in smaller chunks to avoid OOM
    loss_sum = 0.0
    total_samples = 0
    
    for i in range(0, actual_sample_size, chunk_size):
        end_idx = min(i + chunk_size, actual_sample_size)
        xyz_chunk = xyz_sample[i:end_idx]
        albedo_chunk = albedo_sample[i:end_idx]
        
        # Compute pairwise distances for this chunk
        dist_matrix = torch.cdist(xyz_chunk, xyz)  # (chunk_size, N)
        neighbor_mask = dist_matrix < radius  # (chunk_size, N)
        
        # Compute neighborhood average albedo
        neighbor_counts = neighbor_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (chunk_size, 1)
        neighbor_albedo_sum = torch.matmul(neighbor_mask.float(), albedo)  # (chunk_size, 3)
        albedo_mean = neighbor_albedo_sum / neighbor_counts  # (chunk_size, 3)
        
        # L1 smoothness loss for this chunk
        chunk_loss = torch.abs(albedo_chunk - albedo_mean).sum()
        loss_sum += chunk_loss
        total_samples += (end_idx - i)
        
        # Free memory
        del dist_matrix, neighbor_mask, neighbor_counts, neighbor_albedo_sum, albedo_mean
    
    loss = loss_sum / (total_samples * 3)  # Average over all samples and channels
    
    return weight * loss


def residual_energy_loss(gaussians, weight=0.001, threshold=0.1):
    """
    Residual energy regularization.
    Penalizes large residual SH coefficients to encourage the model
    to rely more on the physically-based diffuse component.
    
    Args:
        gaussians: GaussianModel instance
        weight: loss weight
        threshold: energy threshold, only penalize residuals above this
    
    Returns:
        Weighted energy loss
    """
    features_rest = gaussians.get_features_rest  # (N, 15, 3)
    
    if features_rest.shape[0] == 0:
        return torch.tensor(0.0, device="cuda")
    
    # Compute L2 energy per Gaussian
    energy = torch.sum(features_rest ** 2, dim=[1, 2])  # (N,)
    
    # Only penalize residuals above threshold
    energy_penalty = torch.relu(energy - threshold)
    
    return weight * energy_penalty.mean()


def get_residual_weight(iteration, warmup_start=5000, warmup_end=15000):
    """
    Residual weight scheduler for progressive training.
    
    Training stages:
    - [0, warmup_start]: weight = 0 (only learn diffuse component)
    - [warmup_start, warmup_end]: weight = 0→1 (gradually introduce residual)
    - [warmup_end, ∞]: weight = 1 (full residual contribution)
    
    Args:
        iteration: current training iteration
        warmup_start: iteration to start introducing residual
        warmup_end: iteration to reach full residual weight
    
    Returns:
        Residual weight in [0, 1]
    """
    if iteration < warmup_start:
        return 0.0
    elif iteration < warmup_end:
        # Smooth cubic interpolation: 3t^2 - 2t^3
        progress = (iteration - warmup_start) / (warmup_end - warmup_start)
        smooth_progress = 3 * progress**2 - 2 * progress**3
        return smooth_progress
    else:
        return 1.0
