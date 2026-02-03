# Multi-view Chromaticity Consistency Loss

## 功能说明

这是一个为改进版3D Gaussian Splatting添加的多视角色度一致性损失，用于提高反照率（albedo）估计的准确性。

## 原理

漫反射颜色（albedo）是表面的固有属性，与观察视角无关。通过约束不同视角下同一3D点的色度（chromaticity）一致性，可以更好地分离反照率和光照。

### 核心步骤：

1. **双视角渲染**：对主视角和随机选择的副视角分别渲染diffuse-only图像和深度图
2. **几何对应**：使用深度和相机位姿将主视角像素投影到副视角
3. **色度比较**：计算两个视角对应像素的色度差异（RGB归一化后的比例）
4. **遮挡处理**：只在几何匹配有效且无遮挡的像素上计算损失

## 使用方法

### 训练时启用

```bash
python train.py -s <dataset_path> -m <output_path> \
    --enable_chroma_consistency \
    --chroma_lambda 0.01 \
    --chroma_start_iter 3000 \
    --chroma_sample_freq 2 \
    --chroma_depth_threshold 0.01
```

### 参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--enable_chroma_consistency` | False | 是否启用色度一致性损失 |
| `--chroma_lambda` | 0.01 | 色度损失权重 |
| `--chroma_start_iter` | 3000 | 从第几次迭代开始启用（建议等深度稳定后） |
| `--chroma_sample_freq` | 2 | 每几次迭代采样一次（降低计算开销） |
| `--chroma_depth_threshold` | 0.01 | 深度差异阈值（判断遮挡） |

## 实现细节

### 新增文件

- `utils/chroma_utils.py`: 色度一致性损失的核心实现

### 修改文件

1. **arguments/__init__.py**: 添加色度一致性相关参数
2. **gaussian_renderer/__init__.py**: 添加`render_diffuse_only`参数支持
3. **train.py**: 在训练循环中集成色度一致性损失

### 关键函数

```python
# 计算色度
chromaticity, intensity = compute_chromaticity(rgb)

# 视角变换
image_warped, valid_mask, depth_warped = warp_to_other_view(
    depth_src, K_src, K_tgt, T_src_to_tgt, image_src
)

# 遮挡检测
occlusion_mask = compute_occlusion_mask(depth_warped, depth_tgt, threshold)

# 色度一致性损失
loss_chroma, valid_ratio = compute_chroma_consistency_loss(
    camera_src, camera_tgt, gaussians, pipe, background, depth_threshold
)
```

## 预期效果

- ✅ 更准确的反照率估计
- ✅ 更好的光照分离
- ✅ 提高多视角一致性
- ⚠️ 增加约10-20%的训练时间（取决于采样频率）

## 注意事项

1. **深度依赖**：需要渲染的深度图较为准确，建议从3000次迭代后启用
2. **计算开销**：每次采样需要额外渲染一个视角，可通过`chroma_sample_freq`控制频率
3. **场景适用性**：对于有丰富视角重叠的场景效果更好
4. **参数调优**：`chroma_lambda`需要根据具体场景调整，过大可能影响收敛

## 监控指标

训练时会在进度条显示：

- `Chroma Loss`: 色度一致性损失值
- `Valid Ratio`: 有效像素比例（用于判断视角重叠程度）

建议`Valid Ratio`保持在0.3以上，过低说明视角重叠不足。

## 示例输出

```
Training progress: 45%|████▌     | 13500/30000 [15:23<17:42, 15.54it/s, 
    Loss=0.0234567, 
    Depth Loss=0.0012345, 
    Chroma Loss=0.0045678, 
    Valid Ratio=0.456]
```

## 引用

如果这个功能对你的研究有帮助，请引用原始3DGS论文和相关工作。
