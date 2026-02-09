"""
快速测试脚本：验证PBR-GS实现是否正确
"""

import torch
import sys
sys.path.append('.')

def test_imports():
    """测试所有导入是否正常"""
    print("=" * 50)
    print("测试1: 检查导入")
    print("=" * 50)
    
    try:
        from scene.gaussian_model import GaussianModel
        print("✅ GaussianModel 导入成功")
        
        from utils.pbr_utils import cook_torrance_brdf, get_dominant_light_direction
        print("✅ PBR工具函数导入成功")
        
        from arguments import OptimizationParams
        print("✅ 参数配置导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_gaussian_model():
    """测试GaussianModel新增参数"""
    print("\n" + "=" * 50)
    print("测试2: GaussianModel新增参数")
    print("=" * 50)
    
    try:
        from scene.gaussian_model import GaussianModel
        
        # 创建模型
        gaussians = GaussianModel(sh_degree=3)
        print("✅ GaussianModel创建成功")
        
        # 检查新增属性
        assert hasattr(gaussians, '_roughness'), "缺少 _roughness 属性"
        assert hasattr(gaussians, '_metallic'), "缺少 _metallic 属性"
        assert hasattr(gaussians, 'num_probes'), "缺少 num_probes 属性"
        assert hasattr(gaussians, 'probe_positions'), "缺少 probe_positions 属性"
        assert hasattr(gaussians, 'probe_env_sh'), "缺少 probe_env_sh 属性"
        print("✅ 所有新增属性存在")
        
        # 检查新增方法
        assert hasattr(gaussians, 'get_roughness'), "缺少 get_roughness 方法"
        assert hasattr(gaussians, 'get_metallic'), "缺少 get_metallic 方法"
        assert hasattr(gaussians, 'init_light_probes'), "缺少 init_light_probes 方法"
        assert hasattr(gaussians, 'get_spatially_varying_env'), "缺少 get_spatially_varying_env 方法"
        print("✅ 所有新增方法存在")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_light_probes():
    """测试光探针初始化"""
    print("\n" + "=" * 50)
    print("测试3: 光探针初始化")
    print("=" * 50)
    
    try:
        from scene.gaussian_model import GaussianModel
        
        gaussians = GaussianModel(sh_degree=3)
        
        # 初始化光探针
        min_xyz = torch.tensor([-1.0, -1.0, -1.0])
        max_xyz = torch.tensor([1.0, 1.0, 1.0])
        gaussians.init_light_probes((min_xyz, max_xyz), "cpu")
        
        print(f"✅ 光探针数量: {gaussians.num_probes}")
        print(f"✅ 探针位置形状: {gaussians.probe_positions.shape}")
        print(f"✅ 探针环境光形状: {gaussians.probe_env_sh.shape}")
        
        assert gaussians.probe_positions.shape == (16, 3), "探针位置形状错误"
        assert gaussians.probe_env_sh.shape == (16, 3, 25), "探针环境光形状错误"
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spatially_varying_env():
    """测试空间变化环境光插值"""
    print("\n" + "=" * 50)
    print("测试4: 空间变化环境光插值")
    print("=" * 50)
    
    try:
        from scene.gaussian_model import GaussianModel
        
        gaussians = GaussianModel(sh_degree=3)
        
        # 初始化光探针
        min_xyz = torch.tensor([-1.0, -1.0, -1.0])
        max_xyz = torch.tensor([1.0, 1.0, 1.0])
        gaussians.init_light_probes((min_xyz, max_xyz), "cpu")
        
        # 测试插值
        test_positions = torch.randn(100, 3)
        env_sh = gaussians.get_spatially_varying_env(test_positions)
        
        print(f"✅ 输入位置形状: {test_positions.shape}")
        print(f"✅ 输出环境光形状: {env_sh.shape}")
        
        assert env_sh.shape == (100, 3, 25), "环境光插值形状错误"
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cook_torrance_brdf():
    """测试Cook-Torrance BRDF"""
    print("\n" + "=" * 50)
    print("测试5: Cook-Torrance BRDF")
    print("=" * 50)
    
    try:
        from utils.pbr_utils import cook_torrance_brdf
        
        N = 100
        albedo = torch.rand(N, 3)
        normal = torch.randn(N, 3)
        normal = normal / normal.norm(dim=1, keepdim=True)
        view_dir = torch.randn(N, 3)
        view_dir = view_dir / view_dir.norm(dim=1, keepdim=True)
        light_dir = torch.randn(N, 3)
        light_dir = light_dir / light_dir.norm(dim=1, keepdim=True)
        roughness = torch.rand(N, 1)
        metallic = torch.rand(N, 1)
        light_intensity = torch.rand(N, 3)
        
        specular = cook_torrance_brdf(
            albedo, normal, view_dir, light_dir,
            roughness, metallic, light_intensity
        )
        
        print(f"✅ 输入形状: albedo={albedo.shape}, normal={normal.shape}")
        print(f"✅ 输出形状: specular={specular.shape}")
        print(f"✅ 输出范围: [{specular.min():.4f}, {specular.max():.4f}]")
        
        assert specular.shape == (N, 3), "BRDF输出形状错误"
        assert not torch.isnan(specular).any(), "BRDF输出包含NaN"
        assert not torch.isinf(specular).any(), "BRDF输出包含Inf"
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimization_params():
    """测试新增的优化参数"""
    print("\n" + "=" * 50)
    print("测试6: 优化参数")
    print("=" * 50)
    
    try:
        from argparse import ArgumentParser
        from arguments import OptimizationParams
        
        parser = ArgumentParser()
        opt = OptimizationParams(parser)
        
        # 检查新增参数
        assert hasattr(opt, 'roughness_lr'), "缺少 roughness_lr"
        assert hasattr(opt, 'metallic_lr'), "缺少 metallic_lr"
        assert hasattr(opt, 'probe_lr'), "缺少 probe_lr"
        assert hasattr(opt, 'roughness_smooth_weight'), "缺少 roughness_smooth_weight"
        assert hasattr(opt, 'metallic_binary_weight'), "缺少 metallic_binary_weight"
        assert hasattr(opt, 'probe_smooth_weight'), "缺少 probe_smooth_weight"
        
        print(f"✅ roughness_lr = {opt.roughness_lr}")
        print(f"✅ metallic_lr = {opt.metallic_lr}")
        print(f"✅ probe_lr = {opt.probe_lr}")
        print(f"✅ roughness_smooth_weight = {opt.roughness_smooth_weight}")
        print(f"✅ metallic_binary_weight = {opt.metallic_binary_weight}")
        print(f"✅ probe_smooth_weight = {opt.probe_smooth_weight}")
        
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "🚀" * 25)
    print("PBR-GS 实现验证测试")
    print("🚀" * 25 + "\n")
    
    tests = [
        ("导入测试", test_imports),
        ("GaussianModel测试", test_gaussian_model),
        ("光探针初始化测试", test_light_probes),
        ("空间变化环境光测试", test_spatially_varying_env),
        ("Cook-Torrance BRDF测试", test_cook_torrance_brdf),
        ("优化参数测试", test_optimization_params),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} 发生异常: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！PBR-GS实现正确！")
        print("\n下一步:")
        print("1. 准备COLMAP数据集")
        print("2. 运行训练: python train.py -s <dataset_path> -m <output_path>")
        print("3. 查看README_PBR.md了解更多细节")
    else:
        print("\n⚠️ 部分测试失败，请检查实现")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
