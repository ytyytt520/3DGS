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


def albedo_smooth_loss(gaussians, k=16, weight=0.01, sample_size=1024):
    """
    Albedo 3D spatial smoothness regularization (Memory-efficient version).
    Encourages neighboring Gaussians to have similar albedo values.
    
    Args:
        gaussians: GaussianModel instance
        k: number of nearest neighbors to use for smoothing
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
    
    # Use very small sample size to avoid OOM
    actual_sample_size = min(sample_size, N, 512)
    
    # Random sampling for efficiency
    if N > actual_sample_size:
        indices = torch.randperm(N, device=xyz.device)[:actual_sample_size]
        xyz_sample = xyz[indices]
        albedo_sample = albedo[indices]
    else:
        xyz_sample = xyz
        albedo_sample = albedo
        actual_sample_size = N
    
    # Use KNN-based approach instead of radius-based to save memory
    try:
        # Compute k nearest neighbors for each sampled point
        from simple_knn._C import distCUDA2
        
        # Process in very small chunks to avoid OOM
        chunk_size = 128  # Very small chunks
        loss_sum = 0.0
        total_samples = 0
        
        for i in range(0, actual_sample_size, chunk_size):
            end_idx = min(i + chunk_size, actual_sample_size)
            xyz_chunk = xyz_sample[i:end_idx]
            albedo_chunk = albedo_sample[i:end_idx]
            chunk_len = end_idx - i
            
            # Compute distances to all points for this small chunk
            # Use CPU for distance computation if GPU memory is tight
            with torch.no_grad():
                dist_matrix = torch.cdist(xyz_chunk, xyz)  # (chunk_len, N)
                
                # Get k nearest neighbors
                k_actual = min(k, N)
                _, knn_indices = torch.topk(dist_matrix, k_actual, dim=1, largest=False)  # (chunk_len, k)
                
                # Gather albedo values of k nearest neighbors
                knn_albedo = albedo[knn_indices.view(-1)].view(chunk_len, k_actual, 3)  # (chunk_len, k, 3)
                
                # Compute mean albedo of neighbors
                albedo_mean = knn_albedo.mean(dim=1)  # (chunk_len, 3)
            
            # L1 smoothness loss for this chunk
            chunk_loss = torch.abs(albedo_chunk - albedo_mean).sum()
            loss_sum += chunk_loss
            total_samples += chunk_len
            
            # Free memory immediately
            del dist_matrix, knn_indices, knn_albedo, albedo_mean
            torch.cuda.empty_cache()
        
        loss = loss_sum / (total_samples * 3)  # Average over all samples and channels
        
    except Exception as e:
        # Fallback: simple L2 regularization on albedo if KNN fails
        print(f"Warning: albedo_smooth_loss failed ({e}), using simple regularization")
        loss = torch.mean(albedo ** 2) * 0.01
    
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
