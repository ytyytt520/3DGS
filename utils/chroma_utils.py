#
# Optimized Multi-view Chromaticity Consistency Loss
# 优化版：提升效果 + 降低计算开销
#

import torch
import torch.nn.functional as F
from gaussian_renderer import render


def compute_chromaticity(rgb):
    """
    计算色度（归一化的RGB比例）
    
    参数:
        rgb: (C, H, W) 或 (B, C, H, W), RGB图像
    
    返回:
        chromaticity: 同shape, 归一化后的色度
        intensity: (1, H, W) 或 (B, 1, H, W), 亮度
    """
    intensity = rgb.sum(dim=-3, keepdim=True) + 1e-6
    chromaticity = rgb / intensity
    return chromaticity, intensity


def get_intrinsic_matrix(camera):
    """
    从相机对象构建内参矩阵
    
    参数:
        camera: Camera对象
    
    返回:
        K: (3, 3) 内参矩阵
    """
    fovx = camera.FoVx if isinstance(camera.FoVx, float) else camera.FoVx.item() if hasattr(camera.FoVx, 'item') else float(camera.FoVx)
    fovy = camera.FoVy if isinstance(camera.FoVy, float) else camera.FoVy.item() if hasattr(camera.FoVy, 'item') else float(camera.FoVy)
    
    fx = camera.image_width / (2 * torch.tan(torch.tensor(fovx / 2)))
    fy = camera.image_height / (2 * torch.tan(torch.tensor(fovy / 2)))
    cx = camera.image_width / 2.0
    cy = camera.image_height / 2.0
    
    K = torch.tensor([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], device='cuda', dtype=torch.float32)
    
    return K


def sample_random_patch(H, W, patch_size):
    """
    随机采样一个patch区域
    
    返回:
        y_start, y_end, x_start, x_end
    """
    y_start = torch.randint(0, max(1, H - patch_size), (1,)).item()
    x_start = torch.randint(0, max(1, W - patch_size), (1,)).item()
    y_end = min(y_start + patch_size, H)
    x_end = min(x_start + patch_size, W)
    return y_start, y_end, x_start, x_end


def warp_to_other_view_fast(depth_src, K_src, K_tgt, T_src_to_tgt, image_src, use_patch=False, patch_coords=None):
    """
    优化版：将源视角的图像通过深度warp到目标视角
    
    优化点：
    1. 支持patch采样，减少计算量
    2. 使用更高效的张量操作
    """
    C, H, W = image_src.shape
    device = image_src.device
    
    # 如果使用patch，只处理patch区域
    if use_patch and patch_coords is not None:
        y_start, y_end, x_start, x_end = patch_coords
        H_patch = y_end - y_start
        W_patch = x_end - x_start
        
        # 生成patch区域的像素坐标
        y, x = torch.meshgrid(
            torch.arange(y_start, y_end, device=device, dtype=torch.float32),
            torch.arange(x_start, x_end, device=device, dtype=torch.float32),
            indexing='ij'
        )
        depth_src_patch = depth_src[:, y_start:y_end, x_start:x_end]
    else:
        # 全图处理
        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        depth_src_patch = depth_src
        H_patch, W_patch = H, W
    
    # 像素坐标
    pixels_src = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1)
    
    # 反投影到3D
    K_src_inv = torch.inverse(K_src)
    rays = K_src_inv @ pixels_src
    depth_flat = depth_src_patch.reshape(1, -1)
    points_3d_src = rays * depth_flat
    
    # 齐次坐标
    points_3d_src_homo = torch.cat([
        points_3d_src, 
        torch.ones(1, H_patch * W_patch, device=device)
    ], dim=0)
    
    # 变换到目标相机坐标系
    points_3d_tgt = T_src_to_tgt @ points_3d_src_homo
    points_3d_tgt = points_3d_tgt[:3, :]
    
    # 投影到目标视角
    pixels_tgt = K_tgt @ points_3d_tgt
    
    depth_tgt = pixels_tgt[2:3, :]
    pixels_tgt_norm = pixels_tgt[:2, :] / (depth_tgt + 1e-6)
    
    # 重塑
    x_tgt = pixels_tgt_norm[0, :].reshape(H_patch, W_patch)
    y_tgt = pixels_tgt_norm[1, :].reshape(H_patch, W_patch)
    depth_tgt = depth_tgt.reshape(1, H_patch, W_patch)
    
    # 有效掩码
    valid_mask = (
        (x_tgt >= 0) & (x_tgt < W - 1) &
        (y_tgt >= 0) & (y_tgt < H - 1) &
        (depth_tgt[0] > 0)
    ).float().unsqueeze(0)
    
    # Grid sample
    grid_x = 2.0 * x_tgt / (W - 1) - 1.0
    grid_y = 2.0 * y_tgt / (H - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    
    # Warp图像
    if use_patch and patch_coords is not None:
        image_src_patch = image_src[:, y_start:y_end, x_start:x_end].unsqueeze(0)
    else:
        image_src_patch = image_src.unsqueeze(0)
    
    image_warped = F.grid_sample(
        image_src_patch, 
        grid, 
        mode='bilinear', 
        padding_mode='zeros',
        align_corners=True
    ).squeeze(0)
    
    return image_warped, valid_mask, depth_tgt


def compute_occlusion_mask(depth_warped, depth_tgt, threshold=0.03):
    """
    计算遮挡掩码
    """
    depth_diff = torch.abs(depth_warped - depth_tgt)
    relative_diff = depth_diff / (depth_tgt + 1e-6)
    occlusion_mask = (relative_diff < threshold).float()
    return occlusion_mask


def compute_chroma_consistency_loss_optimized(
    camera_src, camera_tgt, 
    gaussians, pipe, background,
    depth_threshold=0.03,
    use_patch=False,
    patch_size=64
):
    """
    优化版：计算多视角色度一致性损失
    
    优化点：
    1. 支持patch采样，大幅减少计算量
    2. 使用no_grad包裹不需要梯度的部分
    3. 提前返回无效情况
    """
    
    # 1. 渲染两个视角的 diffuse-only 图像和深度
    with torch.no_grad():  # 深度和目标图像不需要梯度
        render_tgt = render(
            camera_tgt, gaussians, pipe, background,
            render_diffuse_only=True,
            use_trained_exp=False
        )
        diffuse_tgt = render_tgt["render"]
        depth_tgt = render_tgt["depth"]
    
    # 源视角需要梯度
    render_src = render(
        camera_src, gaussians, pipe, background,
        render_diffuse_only=True,
        use_trained_exp=False
    )
    diffuse_src = render_src["render"]
    depth_src = render_src["depth"]
    
    # 2. 如果使用patch，随机采样一个区域
    patch_coords = None
    if use_patch:
        H, W = diffuse_src.shape[1], diffuse_src.shape[2]
        patch_coords = sample_random_patch(H, W, patch_size)
        y_start, y_end, x_start, x_end = patch_coords
        
        # 裁剪目标图像和深度到patch区域
        diffuse_tgt = diffuse_tgt[:, y_start:y_end, x_start:x_end]
        depth_tgt = depth_tgt[:, y_start:y_end, x_start:x_end]
    
    # 3. 计算相机变换矩阵
    with torch.no_grad():
        T_world_to_src = camera_src.world_view_transform.transpose(0, 1)
        T_world_to_tgt = camera_tgt.world_view_transform.transpose(0, 1)
        T_src_to_tgt = T_world_to_tgt @ torch.inverse(T_world_to_src)
        
        K_src = get_intrinsic_matrix(camera_src)
        K_tgt = get_intrinsic_matrix(camera_tgt)
    
    # 4. Warp源视角到目标视角
    diffuse_src_warped, valid_mask, depth_src_warped = warp_to_other_view_fast(
        depth_src, K_src, K_tgt, T_src_to_tgt, diffuse_src,
        use_patch=use_patch, patch_coords=patch_coords
    )
    
    # 5. 计算遮挡掩码
    with torch.no_grad():
        occlusion_mask = compute_occlusion_mask(
            depth_src_warped, depth_tgt, threshold=depth_threshold
        )
        final_mask = valid_mask * occlusion_mask
        
        # 提前检查有效像素数量
        num_valid = final_mask.sum()
        if num_valid < 100:  # 如果有效像素太少，直接返回0
            return torch.tensor(0.0, device='cuda'), 0.0
    
    # 6. 计算色度
    chroma_src_warped, _ = compute_chromaticity(diffuse_src_warped)
    chroma_tgt, _ = compute_chromaticity(diffuse_tgt)
    
    # 7. 计算色度差异（使用余弦相似度，更鲁棒）
    # L1损失
    chroma_diff_l1 = torch.abs(chroma_src_warped - chroma_tgt)
    
    # 余弦相似度损失（可选，更鲁棒）
    chroma_src_flat = chroma_src_warped.reshape(3, -1)
    chroma_tgt_flat = chroma_tgt.reshape(3, -1)
    cosine_sim = F.cosine_similarity(chroma_src_flat, chroma_tgt_flat, dim=0)
    chroma_diff_cosine = (1 - cosine_sim).reshape(chroma_diff_l1.shape[1:]).unsqueeze(0)
    
    # 组合两种损失
    chroma_diff = 0.7 * chroma_diff_l1 + 0.3 * chroma_diff_cosine
    
    # 8. 加权平均
    masked_diff = chroma_diff * final_mask
    loss = masked_diff.sum() / (num_valid + 1e-6)
    
    valid_ratio = (num_valid / final_mask.numel()).item()
    
    return loss, valid_ratio


def compute_albedo_smoothness_loss(gaussians):
    """
    计算反照率平滑正则化损失
    
    鼓励相邻高斯点的反照率相似
    """
    albedo = gaussians.get_albedo  # (N, 3)
    positions = gaussians.get_xyz   # (N, 3)
    
    # 随机采样一部分点（降低计算量）
    N = albedo.shape[0]
    if N > 10000:
        indices = torch.randperm(N, device='cuda')[:10000]
        albedo_sample = albedo[indices]
        positions_sample = positions[indices]
    else:
        albedo_sample = albedo
        positions_sample = positions
    
    # 计算k近邻（简化版：使用随机采样）
    num_samples = min(1000, albedo_sample.shape[0])
    sample_indices = torch.randperm(albedo_sample.shape[0], device='cuda')[:num_samples]
    
    albedo_samples = albedo_sample[sample_indices]
    pos_samples = positions_sample[sample_indices]
    
    # 计算距离矩阵（只计算采样点之间）
    dist_matrix = torch.cdist(pos_samples, pos_samples)  # (num_samples, num_samples)
    
    # 找到最近的k个邻居（k=5）
    k = 5
    _, nearest_indices = torch.topk(dist_matrix, k=k+1, largest=False, dim=1)
    nearest_indices = nearest_indices[:, 1:]  # 排除自己
    
    # 计算反照率差异
    albedo_neighbors = albedo_samples[nearest_indices]  # (num_samples, k, 3)
    albedo_center = albedo_samples.unsqueeze(1)  # (num_samples, 1, 3)
    
    albedo_diff = torch.abs(albedo_neighbors - albedo_center)
    loss = albedo_diff.mean()
    
    return loss
