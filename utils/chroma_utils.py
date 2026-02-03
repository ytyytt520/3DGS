#
# Multi-view Chromaticity Consistency Loss
# For improved albedo estimation in 3D Gaussian Splatting
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
    # 计算亮度（RGB之和）
    intensity = rgb.sum(dim=-3, keepdim=True) + 1e-6  # 避免除零
    
    # 归一化得到色度
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
    # FoVx和FoVy可能是float或tensor，统一转换
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


def warp_to_other_view(depth_src, K_src, K_tgt, T_src_to_tgt, image_src):
    """
    将源视角的图像通过深度warp到目标视角
    
    参数:
        depth_src: (1, H, W) 源视角深度图
        K_src: (3, 3) 源相机内参矩阵
        K_tgt: (3, 3) 目标相机内参矩阵
        T_src_to_tgt: (4, 4) 从源到目标的变换矩阵
        image_src: (C, H, W) 源视角图像
    
    返回:
        image_warped: (C, H, W) warp到目标视角的图像
        valid_mask: (1, H, W) 有效像素掩码
        depth_warped: (1, H, W) warp后的深度
    """
    C, H, W = image_src.shape
    device = image_src.device
    
    # 1. 生成源视角像素坐标网格
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # 像素坐标 (3, H*W): [x, y, 1]
    pixels_src = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1)
    
    # 2. 反投影到3D空间（源相机坐标系）
    # P_cam = depth * K_inv @ pixels
    K_src_inv = torch.inverse(K_src)
    rays = K_src_inv @ pixels_src  # (3, H*W)
    depth_flat = depth_src.reshape(1, -1)  # (1, H*W)
    points_3d_src = rays * depth_flat  # (3, H*W)
    
    # 转换为齐次坐标
    points_3d_src_homo = torch.cat([
        points_3d_src, 
        torch.ones(1, H*W, device=device)
    ], dim=0)  # (4, H*W)
    
    # 3. 变换到目标相机坐标系
    points_3d_tgt = T_src_to_tgt @ points_3d_src_homo  # (4, H*W)
    points_3d_tgt = points_3d_tgt[:3, :]  # (3, H*W)
    
    # 4. 投影到目标视角像素坐标
    pixels_tgt = K_tgt @ points_3d_tgt  # (3, H*W)
    
    # 归一化
    depth_tgt = pixels_tgt[2:3, :]  # (1, H*W)
    pixels_tgt_norm = pixels_tgt[:2, :] / (depth_tgt + 1e-6)  # (2, H*W)
    
    # 5. 重塑为图像形状
    x_tgt = pixels_tgt_norm[0, :].reshape(H, W)
    y_tgt = pixels_tgt_norm[1, :].reshape(H, W)
    depth_tgt = depth_tgt.reshape(1, H, W)
    
    # 6. 生成有效掩码
    valid_mask = (
        (x_tgt >= 0) & (x_tgt < W - 1) &
        (y_tgt >= 0) & (y_tgt < H - 1) &
        (depth_tgt[0] > 0)  # 深度为正
    ).float().unsqueeze(0)  # (1, H, W)
    
    # 7. 使用grid_sample进行双线性插值
    # 归一化到[-1, 1]
    grid_x = 2.0 * x_tgt / (W - 1) - 1.0
    grid_y = 2.0 * y_tgt / (H - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    
    # warp图像
    image_src_batch = image_src.unsqueeze(0)  # (1, C, H, W)
    image_warped = F.grid_sample(
        image_src_batch, 
        grid, 
        mode='bilinear', 
        padding_mode='zeros',
        align_corners=True
    ).squeeze(0)  # (C, H, W)
    
    return image_warped, valid_mask, depth_tgt


def compute_occlusion_mask(depth_warped, depth_tgt, threshold=0.01):
    """
    计算遮挡掩码（检查warp后的深度与目标深度是否一致）
    
    参数:
        depth_warped: (1, H, W) warp后的深度
        depth_tgt: (1, H, W) 目标视角渲染的深度
        threshold: float, 深度差异阈值
    
    返回:
        occlusion_mask: (1, H, W) 非遮挡区域为1
    """
    depth_diff = torch.abs(depth_warped - depth_tgt)
    
    # 相对深度差异
    relative_diff = depth_diff / (depth_tgt + 1e-6)
    
    occlusion_mask = (relative_diff < threshold).float()
    
    return occlusion_mask


def compute_chroma_consistency_loss(
    camera_src, camera_tgt, 
    gaussians, pipe, background,
    depth_threshold=0.01
):
    """
    计算多视角色度一致性损失
    
    参数:
        camera_src: Camera, 主视角相机
        camera_tgt: Camera, 副视角相机
        gaussians: GaussianModel
        pipe: PipelineParams
        background: Tensor, 背景颜色
        depth_threshold: float, 深度差异阈值
    
    返回:
        loss: Tensor, 色度一致性损失
        valid_ratio: float, 有效像素比例（用于监控）
    """
    
    # 1. 渲染两个视角的 diffuse-only 图像和深度
    render_src = render(
        camera_src, gaussians, pipe, background,
        render_diffuse_only=True,
        use_trained_exp=False  # 不使用曝光校正
    )
    render_tgt = render(
        camera_tgt, gaussians, pipe, background,
        render_diffuse_only=True,
        use_trained_exp=False
    )
    
    diffuse_src = render_src["render"]  # (3, H, W)
    depth_src = render_src["depth"]     # (1, H, W)
    diffuse_tgt = render_tgt["render"]  # (3, H, W)
    depth_tgt = render_tgt["depth"]     # (1, H, W)
    
    # 2. 计算相机变换矩阵
    # T_src_to_tgt: 从源相机坐标系到目标相机坐标系
    T_world_to_src = camera_src.world_view_transform.transpose(0, 1)  # (4, 4)
    T_world_to_tgt = camera_tgt.world_view_transform.transpose(0, 1)  # (4, 4)
    T_src_to_tgt = T_world_to_tgt @ torch.inverse(T_world_to_src)
    
    # 3. 构建内参矩阵
    K_src = get_intrinsic_matrix(camera_src)  # (3, 3)
    K_tgt = get_intrinsic_matrix(camera_tgt)  # (3, 3)
    
    # 4. Warp源视角到目标视角
    diffuse_src_warped, valid_mask, depth_src_warped = warp_to_other_view(
        depth_src, K_src, K_tgt, T_src_to_tgt, diffuse_src
    )
    
    # 5. 计算遮挡掩码
    occlusion_mask = compute_occlusion_mask(
        depth_src_warped, depth_tgt, threshold=depth_threshold
    )
    
    # 6. 综合掩码
    final_mask = valid_mask * occlusion_mask  # (1, H, W)
    
    # 7. 计算色度
    chroma_src_warped, _ = compute_chromaticity(diffuse_src_warped)
    chroma_tgt, _ = compute_chromaticity(diffuse_tgt)
    
    # 8. 计算色度差异（只在有效区域）
    chroma_diff = torch.abs(chroma_src_warped - chroma_tgt)  # (3, H, W)
    
    # 9. 加权平均
    masked_diff = chroma_diff * final_mask  # (3, H, W)
    
    num_valid = final_mask.sum() + 1e-6
    loss = masked_diff.sum() / num_valid
    
    valid_ratio = (num_valid / final_mask.numel()).item()
    
    return loss, valid_ratio
