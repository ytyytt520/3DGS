"""
⭐ 物理渲染工具函数
Physical-Based Rendering (PBR) utilities for Gaussian Splatting
"""

import torch
import math

def cook_torrance_brdf(albedo, normal, view_dir, light_dir, roughness, metallic, light_intensity):
    """
    Cook-Torrance BRDF模型
    
    参数:
        albedo: (N, 3) 基础颜色/反照率
        normal: (N, 3) 表面法线（归一化）
        view_dir: (N, 3) 观察方向（从表面指向相机，归一化）
        light_dir: (N, 3) 光源方向（从表面指向光源，归一化）
        roughness: (N, 1) 粗糙度 [0,1]
        metallic: (N, 1) 金属度 [0,1]
        light_intensity: (N, 3) 光照强度
    
    返回:
        specular: (N, 3) 镜面反射颜色
    """
    
    # ===== 1. 计算基础向量 =====
    H = torch.nn.functional.normalize(view_dir + light_dir, dim=-1, eps=1e-6)  # 半程向量
    
    NdotH = torch.clamp(torch.sum(normal * H, dim=-1, keepdim=True), 0.0, 1.0)
    NdotV = torch.clamp(torch.sum(normal * view_dir, dim=-1, keepdim=True), 0.0, 1.0)
    NdotL = torch.clamp(torch.sum(normal * light_dir, dim=-1, keepdim=True), 0.0, 1.0)
    VdotH = torch.clamp(torch.sum(view_dir * H, dim=-1, keepdim=True), 0.0, 1.0)
    
    # ===== 2. 法线分布函数 D (GGX/Trowbridge-Reitz) =====
    alpha = roughness ** 2
    alpha2 = alpha ** 2
    denom = NdotH ** 2 * (alpha2 - 1.0) + 1.0
    D = alpha2 / (math.pi * denom ** 2 + 1e-8)  # (N, 1)
    
    # ===== 3. 几何遮蔽函数 G (Smith-GGX) =====
    def smith_g1(NdotX, alpha):
        k = alpha / 2.0
        return NdotX / (NdotX * (1.0 - k) + k + 1e-8)
    
    G = smith_g1(NdotV, alpha) * smith_g1(NdotL, alpha)  # (N, 1)
    
    # ===== 4. 菲涅尔项 F (Schlick近似) =====
    # F0: 0度入射的反射率
    F0 = torch.lerp(
        torch.full_like(albedo, 0.04),  # 非金属 ~4%
        albedo,                          # 金属用albedo作为F0
        metallic
    )  # (N, 3)
    
    F = F0 + (1.0 - F0) * torch.pow(1.0 - VdotH, 5.0)  # (N, 3)
    
    # ===== 5. Cook-Torrance镜面项 =====
    numerator = D * G * F  # (N, 3)
    denominator = 4.0 * NdotV * NdotL + 1e-8
    specular = numerator / denominator  # (N, 3)
    
    # ===== 6. 乘以光照强度和Lambert余弦 =====
    specular = specular * light_intensity * NdotL
    
    return specular


def get_dominant_light_direction(env_sh):
    """
    从环境光球谐系数中提取主光源方向
    
    参数:
        env_sh: (N, 3, K) 环境光球谐系数
    
    返回:
        light_dir: (N, 3) 主光源方向（归一化）
    """
    # 使用1阶球谐系数（索引1,2,3）来估计主光源方向
    # Y1 = -C1 * y, Y2 = C1 * z, Y3 = -C1 * x
    C1 = 0.4886025119029199
    
    # 取RGB三通道的平均
    sh_1 = env_sh[:, :, 1].mean(dim=1, keepdim=True)  # (N, 1) Y方向
    sh_2 = env_sh[:, :, 2].mean(dim=1, keepdim=True)  # (N, 1) Z方向
    sh_3 = env_sh[:, :, 3].mean(dim=1, keepdim=True)  # (N, 1) X方向
    
    # 反推方向
    light_dir = torch.cat([
        -sh_3 / C1,  # x
        -sh_1 / C1,  # y
        sh_2 / C1    # z
    ], dim=-1)  # (N, 3)
    
    # 归一化
    light_dir = torch.nn.functional.normalize(light_dir, dim=-1, eps=1e-6)
    
    return light_dir


def sample_env_for_specular(env_sh, reflect_dir, roughness):
    """
    根据粗糙度从环境光采样镜面反射
    
    参数:
        env_sh: (N, 3, K) 环境光球谐系数
        reflect_dir: (N, 3) 反射方向
        roughness: (N, 1) 粗糙度
    
    返回:
        env_specular: (N, 3) 环境镜面反射
    """
    from utils.sh_utils import eval_sh
    
    # 粗糙度越大，使用越低阶的球谐（模糊效果）
    # roughness=0 -> degree=4, roughness=1 -> degree=0
    degree = 4 - int(roughness.mean().item() * 4)
    degree = max(0, min(4, degree))
    
    # 从环境光采样
    env_specular = torch.relu(eval_sh(degree, env_sh, reflect_dir))
    
    return env_specular


def compute_material_regularization(gaussians, opt):
    """
    计算材质正则化损失
    
    参数:
        gaussians: GaussianModel
        opt: OptimizationParams
    
    返回:
        loss_dict: 包含各项正则化损失的字典
    """
    losses = {}
    
    # ===== 1. 粗糙度平滑正则化 =====
    # 相邻高斯的粗糙度应该相似
    roughness = gaussians.get_roughness  # (N, 1)
    positions = gaussians.get_xyz  # (N, 3)
    
    # 简化版：计算随机采样点对的粗糙度差异
    N = positions.shape[0]
    if N > 1000:
        # 随机采样1000个点
        indices = torch.randperm(N, device=positions.device)[:1000]
        sample_pos = positions[indices]
        sample_rough = roughness[indices]
        
        # 找最近邻
        distances = torch.cdist(sample_pos, positions)  # (1000, N)
        nearest_idx = distances.topk(k=5, dim=1, largest=False)[1]  # (1000, 5)
        
        # 计算粗糙度差异
        nearest_rough = roughness[nearest_idx]  # (1000, 5, 1)
        rough_diff = (sample_rough.unsqueeze(1) - nearest_rough) ** 2
        losses['roughness_smooth'] = opt.roughness_smooth_weight * rough_diff.mean()
    else:
        losses['roughness_smooth'] = torch.tensor(0.0, device=positions.device)
    
    # ===== 2. 金属度二值化正则化 =====
    # 鼓励金属度接近0或1
    metallic = gaussians.get_metallic  # (N, 1)
    losses['metallic_binary'] = opt.metallic_binary_weight * torch.mean(metallic * (1.0 - metallic))
    
    # ===== 3. 光探针平滑正则化 =====
    # 相邻探针的环境光应该相似
    probe_positions = gaussians.probe_positions  # (K, 3)
    probe_env_sh = gaussians.probe_env_sh  # (K, 3, 25)
    
    K = probe_positions.shape[0]
    if K > 1:
        # 计算探针之间的距离
        probe_distances = torch.cdist(probe_positions, probe_positions)  # (K, K)
        
        # 找相邻探针（距离小于阈值）
        threshold = (probe_positions.max() - probe_positions.min()) / 3.0
        neighbors = (probe_distances < threshold).float()  # (K, K)
        neighbors = neighbors - torch.eye(K, device=neighbors.device)  # 去掉自己
        
        # 计算相邻探针的环境光差异
        env_diff = probe_env_sh.unsqueeze(0) - probe_env_sh.unsqueeze(1)  # (K, K, 3, 25)
        env_diff_sq = (env_diff ** 2).sum(dim=[2, 3])  # (K, K)
        
        losses['probe_smooth'] = opt.probe_smooth_weight * (env_diff_sq * neighbors).sum() / (neighbors.sum() + 1e-8)
    else:
        losses['probe_smooth'] = torch.tensor(0.0, device=probe_positions.device)
    
    return losses
