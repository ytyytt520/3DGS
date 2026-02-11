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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.pbr_utils import cook_torrance_brdf, get_dominant_light_direction, sample_env_for_specular

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # ⭐ 物理渲染：计算颜色
    shs = None
    dc = None
    colors_precomp = None
    if override_color is None:
        num_pts = pc.get_xyz.shape[0]
        
        # ===== 1. 获取材质参数 =====
        albedo = pc.get_albedo  # (N, 3)
        normals = pc.get_normal  # (N, 3)
        roughness = pc.get_roughness  # (N, 1)
        metallic = pc.get_metallic  # (N, 1)
        
        # ===== 2. 获取空间变化的环境光 ⭐ =====
        positions = pc.get_xyz  # (N, 3)
        env_sh = pc.get_spatially_varying_env(positions)  # (N, 3, 9) ⭐ 降低到9系数
        
        # ===== 3. 计算漫反射 =====
        diffuse_light = torch.relu(eval_sh(2, env_sh, normals))  # ⭐ 使用2阶球谐（节省显存）
        # 能量守恒：金属没有漫反射
        diffuse_color = albedo * diffuse_light * (1.0 - metallic)  # (N, 3)
        
        # ===== 4. 计算镜面反射 ⭐ =====
        specular_color = torch.zeros_like(albedo)
        
        if not pipe.diffuse_only:
            # 计算观察方向（从表面指向相机）
            view_dirs = viewpoint_camera.camera_center.repeat(num_pts, 1) - pc.get_xyz
            view_dirs = view_dirs / (view_dirs.norm(dim=1, keepdim=True) + 1e-9)
            
            # 获取主光源方向
            light_dirs = get_dominant_light_direction(env_sh)  # (N, 3)
            
            # 计算光照强度
            light_intensity = torch.relu(eval_sh(2, env_sh, light_dirs))  # ⭐ 使用2阶球谐
            
            # Cook-Torrance BRDF
            specular_direct = cook_torrance_brdf(
                albedo=albedo,
                normal=normals,
                view_dir=view_dirs,
                light_dir=light_dirs,
                roughness=roughness,
                metallic=metallic,
                light_intensity=light_intensity
            )  # (N, 3)
            
            # 环境镜面反射（简化版，减少计算）
            reflect_dirs = 2.0 * torch.sum(view_dirs * normals, dim=-1, keepdim=True) * normals - view_dirs
            reflect_dirs = reflect_dirs / (reflect_dirs.norm(dim=1, keepdim=True) + 1e-9)
            env_specular = sample_env_for_specular(env_sh, reflect_dirs, roughness)  # (N, 3)
            
            # 菲涅尔加权
            VdotN = torch.clamp(torch.sum(view_dirs * normals, dim=-1, keepdim=True), 0.0, 1.0)
            F0 = torch.lerp(torch.full_like(albedo, 0.04), albedo, metallic)
            F = F0 + (1.0 - F0) * torch.pow(1.0 - VdotN, 5.0)
            
            specular_env = F * env_specular
            
            specular_color = specular_direct + specular_env
            
            # ===== 5. 可选：球谐残差（处理极端情况） =====
            residual_sh = torch.cat((torch.zeros_like(pc.get_features_dc), pc.get_features_rest), dim=1).transpose(1, 2)
            residual_color = 0.05 * eval_sh(pc.active_sh_degree, residual_sh, view_dirs)  # ⭐ 进一步降低权重
            specular_color = specular_color + residual_color
        
        # ===== 6. 合成最终颜色 =====
        colors_precomp = torch.clamp(diffuse_color + specular_color, 0.0, 1.0)
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out
