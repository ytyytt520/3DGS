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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB, C0
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            #构建缩放缩放矩阵
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            #计算协方差矩阵
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        self.albedo_activation = torch.sigmoid
        self.normal_activation = lambda n: torch.nn.functional.normalize(n, dim=1, eps=1e-6)


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0 #激活的球谐函数阶数
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  #最大球谐函数阶数
        self._xyz = torch.empty(0) #位置
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._albedo = torch.empty(0)
        self._normal = torch.empty(0)
        self._scaling = torch.empty(0) #尺度因子
        self._rotation = torch.empty(0) #旋转向量
        self._opacity = torch.empty(0) #不透明度
        self._env_sh = torch.empty(0)
        self.env_sh_degree = 4  # ⭐ 提高到4阶（25个系数）
        self.env_sh_dc_anchor = 1.0 / C0
        
        # ⭐ 新增：物理材质参数
        self._roughness = torch.empty(0)  # 粗糙度
        self._metallic = torch.empty(0)   # 金属度
        
        # ⭐ 新增：空间变化环境光（光探针）
        self.num_probes = 16  # 探针数量
        self.probe_positions = torch.empty(0)  # 探针位置
        self.probe_env_sh = torch.empty(0)     # 每个探针的环境光
        
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._albedo,
            self._normal,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._env_sh,
        )
    
    def restore(self, model_args, training_args):
        if len(model_args) < 15:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
            device = self._xyz.device
            default_albedo = torch.full((self._xyz.shape[0], 3), 0.5, device=device)
            self._albedo = nn.Parameter(self.inverse_opacity_activation(default_albedo).requires_grad_(True))
            self._normal = nn.Parameter(self._default_normals(device, self._xyz.shape[0]).requires_grad_(True))
            self._env_sh = self.init_env_sh(device)
        else:
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._albedo,
            self._normal,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            self._env_sh) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        try:
            self.optimizer.load_state_dict(opt_dict)
        except Exception as e:
            print(f"Warning: optimizer state could not be fully restored ({e}), using freshly initialized optimizer.")

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_albedo(self):
        return self.albedo_activation(self._albedo)
    
    @property
    def get_normal(self):
        return self.normal_activation(self._normal)

    @property
    def get_env_sh(self):
        return self._env_sh
    
    @property
    def get_roughness(self):
        """粗糙度：sigmoid激活到[0,1]"""
        return self.opacity_activation(self._roughness)
    
    @property
    def get_metallic(self):
        """金属度：sigmoid激活到[0,1]"""
        return self.opacity_activation(self._metallic)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def init_env_sh(self, device):
        """初始化全局环境光（用于兼容性，实际使用探针）"""
        env = torch.zeros((1, 3, (self.env_sh_degree + 1) ** 2), device=device, dtype=torch.float)
        env[..., 0] = self.env_sh_dc_anchor
        return nn.Parameter(env.requires_grad_(True))
    
    def init_light_probes(self, scene_bounds, device):
        """
        ⭐ 初始化空间变化的环境光探针
        
        参数:
            scene_bounds: (min_xyz, max_xyz) 场景边界
            device: 设备
        """
        min_xyz, max_xyz = scene_bounds
        
        # 在3D网格中均匀放置探针
        grid_size = int(np.ceil(self.num_probes ** (1/3)))  # 16 -> 3x3x3
        
        x = torch.linspace(min_xyz[0].item(), max_xyz[0].item(), grid_size, device=device)
        y = torch.linspace(min_xyz[1].item(), max_xyz[1].item(), grid_size, device=device)
        z = torch.linspace(min_xyz[2].item(), max_xyz[2].item(), grid_size, device=device)
        
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        
        positions = torch.stack([
            grid_x.flatten(),
            grid_y.flatten(),
            grid_z.flatten()
        ], dim=-1)[:self.num_probes]  # (num_probes, 3)
        
        self.probe_positions = nn.Parameter(positions)
        
        # 初始化每个探针的环境光为均匀白光
        num_coeffs = (self.env_sh_degree + 1) ** 2  # 4阶 = 25个系数
        probe_env = torch.zeros((self.num_probes, 3, num_coeffs), device=device, dtype=torch.float)
        probe_env[:, :, 0] = self.env_sh_dc_anchor  # DC分量
        
        self.probe_env_sh = nn.Parameter(probe_env)
        
        print(f"✅ 初始化 {self.num_probes} 个环境光探针，每个探针 {num_coeffs} 个球谐系数")
    
    def get_spatially_varying_env(self, positions):
        """
        ⭐ 获取空间变化的环境光
        
        参数:
            positions: (N, 3) 查询位置
        返回:
            env_sh: (N, 3, 25) 每个位置的环境光球谐系数
        """
        N = positions.shape[0]
        K = self.num_probes
        
        # 计算每个点到所有探针的距离
        distances = torch.cdist(positions, self.probe_positions)  # (N, K)
        
        # 使用RBF插值（径向基函数）
        sigma = (self.probe_positions.max() - self.probe_positions.min()) / 4.0  # 自适应sigma
        weights = torch.exp(-distances ** 2 / (2 * sigma ** 2 + 1e-8))  # (N, K)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # 归一化
        
        # 加权求和
        env_sh = torch.einsum('nk,kcd->ncd', weights, self.probe_env_sh)  # (N, 3, 25)
        
        return env_sh

    def load_env_sh_tensor(self, env_tensor):
        device = self._xyz.device if self._xyz.numel() > 0 else env_tensor.device
        self._env_sh = nn.Parameter(env_tensor.to(device).requires_grad_(True))

    def _default_normals(self, device, count):
        normals = torch.zeros((count, 3), device=device)
        normals[:, 2] = 1.0
        return normals

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

        normals_np = np.asarray(pcd.normals) if pcd.normals is not None else np.zeros_like(pcd.points)
        if normals_np.shape[0] != fused_point_cloud.shape[0]:
            normals_np = np.zeros_like(pcd.points)
        normals = torch.tensor(normals_np, dtype=torch.float, device="cuda")
        default_normals = self._default_normals("cuda", fused_point_cloud.shape[0])
        normals = torch.where(torch.isnan(normals), default_normals, normals)
        normals = torch.where(normals.abs().sum(dim=1, keepdim=True) < 1e-6, default_normals, normals)
        self._normal = nn.Parameter(normals.requires_grad_(True))

        albedo = torch.tensor(np.asarray(pcd.colors), dtype=torch.float, device="cuda")
        albedo = torch.clamp(albedo, 1e-3, 1.0 - 1e-3)
        self._albedo = nn.Parameter(self.inverse_opacity_activation(albedo).requires_grad_(True))

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        # ⭐ 初始化粗糙度（默认0.5，中等粗糙）
        roughness_init = self.inverse_opacity_activation(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self._roughness = nn.Parameter(roughness_init.requires_grad_(True))
        
        # ⭐ 初始化金属度（默认0.1，大部分是非金属）
        metallic_init = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self._metallic = nn.Parameter(metallic_init.requires_grad_(True))
        
        # ⭐ 初始化空间变化环境光探针
        # 计算场景边界
        min_xyz = fused_point_cloud.min(dim=0)[0]
        max_xyz = fused_point_cloud.max(dim=0)[0]
        self.init_light_probes((min_xyz, max_xyz), "cuda")
        
        # 保留全局环境光用于兼容性
        self._env_sh = self.init_env_sh("cuda")
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._albedo], 'lr': training_args.albedo_lr, "name": "albedo"},
            {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._env_sh], 'lr': training_args.env_lr, "name": "env_sh"},
            # ⭐ 新增：物理材质参数
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
            {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},
            # ⭐ 新增：光探针参数
            {'params': [self.probe_positions], 'lr': training_args.probe_lr, "name": "probe_pos"},
            {'params': [self.probe_env_sh], 'lr': training_args.env_lr, "name": "probe_env"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'albedo_r', 'albedo_g', 'albedo_b']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = self.get_normal.detach().cpu().numpy()
        albedo = self.get_albedo.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, albedo, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        ply_props = plydata.elements[0].data.dtype.names

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if "albedo_r" in ply_props:
            albedo = np.stack((np.asarray(plydata.elements[0]["albedo_r"]),
                               np.asarray(plydata.elements[0]["albedo_g"]),
                               np.asarray(plydata.elements[0]["albedo_b"])), axis=1)
        else:
            albedo = SH2RGB(torch.tensor(features_dc[:, :, 0], dtype=torch.float)).cpu().numpy()
        albedo = np.clip(albedo, 1e-3, 1.0 - 1e-3)

        if "nx" in ply_props and "ny" in ply_props and "nz" in ply_props:
            normals = np.stack((np.asarray(plydata.elements[0]["nx"]),
                                np.asarray(plydata.elements[0]["ny"]),
                                np.asarray(plydata.elements[0]["nz"])), axis=1)
        else:
            normals = self._default_normals("cpu", xyz.shape[0]).cpu().numpy()

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._albedo = nn.Parameter(self.inverse_opacity_activation(torch.tensor(albedo, dtype=torch.float, device="cuda")).requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normals, dtype=torch.float, device="cuda").requires_grad_(True))
        self._env_sh = self.init_env_sh("cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in ["xyz", "f_dc", "f_rest", "albedo", "normal", "opacity", "scaling", "rotation", "roughness", "metallic"]:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # ⭐ 修复：对于不同形状的参数，正确应用mask
                param = group["params"][0]
                if param.ndim == 1:
                    # 1D参数 (N,)
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    new_param = param[mask]
                elif param.ndim == 2 and param.shape[1] == 1:
                    # 2D参数但第二维是1 (N, 1) - 如 opacity, roughness, metallic
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    new_param = param[mask]
                else:
                    # 其他多维参数 (N, D)
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    new_param = param[mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(new_param.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._albedo = optimizable_tensors["albedo"]
        self._normal = optimizable_tensors["normal"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # ⭐ 剪枝材质参数
        if "roughness" in optimizable_tensors:
            self._roughness = optimizable_tensors["roughness"]
        if "metallic" in optimizable_tensors:
            self._metallic = optimizable_tensors["metallic"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] not in tensors_dict:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_albedo, new_normal, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "albedo": new_albedo,
        "normal": new_normal,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._albedo = optimizable_tensors["albedo"]
        self._normal = optimizable_tensors["normal"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_albedo = self._albedo[selected_pts_mask].repeat(N,1)
        new_normal = self._normal[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        
        # ⭐ 复制材质参数
        new_roughness = self._roughness[selected_pts_mask].repeat(N,1)
        new_metallic = self._metallic[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_albedo, new_normal, new_opacity, new_scaling, new_rotation, new_tmp_radii)
        
        # ⭐ 手动添加材质参数（因为densification_postfix不包含它们）
        self._roughness = torch.cat([self._roughness, new_roughness], dim=0)
        self._metallic = torch.cat([self._metallic, new_metallic], dim=0)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_albedo = self._albedo[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        
        # ⭐ 克隆材质参数
        new_roughness = self._roughness[selected_pts_mask]
        new_metallic = self._metallic[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_albedo, new_normal, new_opacities, new_scaling, new_rotation, new_tmp_radii)
        
        # ⭐ 手动添加材质参数
        self._roughness = torch.cat([self._roughness, new_roughness], dim=0)
        self._metallic = torch.cat([self._metallic, new_metallic], dim=0)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
