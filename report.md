# Guided Image Filtering 论文复现总结报告

## 1. 项目背景与目标

本仓库旨在对 Kaiming He、Jian Sun、Xiaoou Tang 等人在 ECCV 2010 及其后续 TPAMI 2013 上发表的经典论文《Guided Image Filtering》进行可重复、可验证的代码级复现。目标包括：

1. 依据论文中的数学推导实现导引滤波器的核心算法；
2. 使用纯 Python 与最小依赖（NumPy、OpenCV、Pillow、Matplotlib）构建可直接运行的示例脚本；
3. 在真实图像上复现论文中的三个代表性应用：边缘保留平滑、细节增强、联合上采样；
4. 输出全套实验结果、参数记录与复现报告，为后续撰写技术总结提供基础资料。

整个项目在 MacBook 环境下使用 `uv` 作为包管理与运行入口，确保脚本执行流程统一。与论文原始实现（Matlab）相比，当前复现保持了相同的数学结构，但使用 Python 生态重写算法，便于现代团队维护。

## 2. 论文要点回顾

### 2.1 核心思想

导引滤波器（Guided Filter）提出一种基于局部线性模型的滤波方案。对任一局部窗口 `omega_k`，假设输出像素 `q_i` 可表示为导引图 `I_i` 的线性函数：

```
q_i = a_k * I_i + b_k, for all i in omega_k
```

其中 `a_k`、`b_k` 在窗口内为常数。通过最小化以下目标函数求得最优线性系数：

```
E(a_k, b_k) = sum_{i in omega_k} [ (a_k * I_i + b_k - p_i)^2 + eps * a_k^2 ]
```

该公式中：

- `p_i` 为待滤波输入；
- `I_i` 为导引图，可与 `p_i` 相同或不同；
- `eps` 为正则化项；
- `a_k`、`b_k` 的闭式解可由局部均值与协方差得到。

最终输出 `q_i` 为与像素 `i` 相邻所有窗口的 `a_k`、`b_k` 平均：

```
q_i = mean_k(a_k) * I_i + mean_k(b_k)
```

### 2.2 相比双边滤波的优势

论文指出导引滤波与双边滤波均能实现边缘保留，但导引滤波具有以下优势：

- 解析解避免了双边滤波在强边界附近的梯度反转（halo）问题；
- 算法时间复杂度 `O(N)`，与核大小无关；
- 能够处理联合滤波任务，如使用高分辨率导引图提升低分辨率输入的上采样效果；
- 与 matting Laplacian 存在理论联系，可扩展到图像抠图、去雾等任务。

## 3. 代码实现概述

复现代码位于 `src/` 目录，辅助脚本位于 `scripts/`。关键模块如下：

1. `src/guided_filter.py`：实现 `guided_filter` 主函数以及灰度/彩色两套内部逻辑。
2. `src/utils.py`：封装图像读取（RGB 浮点归一化）、保存（自动创建目录，支持 0-1 与 0-255 数据）、上采样、参数 JSON 记录等工具函数。
3. `src/demo_smoothing.py`：单图像平滑示例，支持命令行参数配置半径与正则项，输出滤波结果及原图对比。
4. `src/demo_enhance.py`：细节增强示例，提供 `alpha` 调节项，输出增强结果与对比图。
5. `src/demo_joint_upsample.py`：联合上采样示例，接收低分辨率输入与高分辨率导引，生成导引滤波结果与 baseline 对比。
6. `scripts/run_suite.py`：批量运行脚本。自动遍历 `data/input/portrait/`、`data/input/landscape/` 中的图像，按参数网格执行三类实验，并保存结果与参数日志。

### 3.1 核心算法实现细节

`guided_filter.py` 包含以下关键函数：

- `_box_filter(src, radius)`：封装 `cv2.boxFilter`，使用反射边界。确保均值计算与论文中积分图思路一致。
- `guided_filter(I, p, radius, eps)`：根据导引图维度选择灰度或彩色路径。
- `_guided_filter_gray(...)`：按照论文公式计算 `mean_I`、`mean_p`、`mean_Ip`、`var_I`，继而得到 `a`、`b` 与最终输出。
- `_guided_filter_color(...)`：针对彩色导引图的 3×3 协方差矩阵，使用 `numpy.linalg.inv` 与 `einsum` 矢量化求解局部线性系数；同样通过盒滤波获取 `mean_a`、`mean_b`。

实现中所有数组均转换为 `float32`，确保数值稳定；`eps` 添加为正定矩阵以避免奇异矩阵带来的求逆问题。

### 3.2 批量脚本功能

`scripts/run_suite.py` 主要职责：

- 收集输入：默认人像图在 `data/input/portrait/`，风景图在 `data/input/landscape/`。
- `run_smoothing` 函数：对每张图遍历 `radius`、`eps` 组合，输出滤波图与横向拼接对比图，并写入 JSON（字段包括 `input`, `radius`, `eps`, `mode`）。
- `run_enhancement` 函数：先生成基图 `base = guided_filter(image, image, radius, eps)`，然后计算 `enhanced = clip(image + alpha * (image - base))`。输出增强图、对比图及参数 JSON。
- `run_joint_upsample` 函数：将原图缩小（默认 scale=0.25），再用给定方法上采样，与导引滤波结果做对比；参数日志记录 `scale`, `radius`, `eps`, `upsample_method`, `baseline_method`。
- 命令行参数允许跳过特定阶段（`--skip joint` 等），方便阶段性调试。

## 4. 数据准备

实验使用 Kodak 数据集的四张图片，分别放置在指定目录：

- 人像类：`data/input/portrait/kodim04.png`、`data/input/portrait/kodim15.png`
- 风景与结构类：`data/input/landscape/kodim19.png`、`data/input/landscape/kodim22.png`

导入后目录结构如下：

```
data/
  input/
    portrait/
      kodim04.png
      kodim15.png
    landscape/
      kodim19.png
      kodim22.png
  results/
    enhance/         # 由脚本自动生成
    joint_upsample/  # 由脚本自动生成
    smoothing/       # 由脚本自动生成
```

选择这些图片是因为其包含不同类型的边缘与纹理：kodim04/kodim15 为人物及复杂背景，kodim19/kodim22 包含自然景物与人工结构，能充分展示导引滤波在不同场景下的表现。

## 5. 实验流程

### 5.1 运行命令

所有实验通过以下命令一键执行：

```
uv run python scripts/run_suite.py
```

可使用参数定制流程，例如：

```
uv run python scripts/run_suite.py --skip joint
uv run python scripts/run_suite.py --smoothing-radii 4 8 --smoothing-eps 1e-4 1e-3
uv run python scripts/run_suite.py --enhance-alphas 1.2 1.5 1.8
```

### 5.2 参数网格

1. 平滑实验参数：
   - 半径 `radius`: 4, 8, 16
   - 正则项 `eps`: 1e-4, 1e-3, 1e-2
   - 图像：4 张
   - 总组合：4 (图像) × 3 (半径) × 3 (eps) = 36

2. 细节增强参数：
   - `alpha`: 1.0, 1.5, 2.0
   - `radius`: 8
   - `eps`: 1e-3
   - 图像：2 张（仅人像）
   - 总组合：2 × 3 = 6

3. 联合上采样参数：
   - `scale`: 0.25
   - `upsample_method`: nearest
   - `baseline_method`: bilinear
   - `radius`: 4
   - `eps`: 1e-4
   - 图像：2 张（风景）
   - 总组合：2

实验过程中，脚本自动创建 `data/results/<mode>/` 子目录，并生成 `.png` 与 `.json` 文件。

## 6. 实验结果与分析

### 6.1 平滑实验

生成的结果位于 `data/results/smoothing/`，命名格式为：

```
<stem>_r<radius>_eps<eps>.png
<stem>_r<radius>_eps<eps>_comparison.png
<stem>_r<radius>_eps<eps>.json
```

#### 6.1.1 `kodim04`（人像）

- `data/results/smoothing/kodim04_r4_eps1e-04_comparison.png` 展示在 `radius=4`, `eps=1e-4` 时的效果：皮肤纹理被适度平滑，头发边缘保持清晰。
- 当半径增大到 16，`eps` 增大到 `1e-2`（`kodim04_r16_eps1e-02_comparison.png`）时，背景被显著平滑，人物轮廓仍保持锐利，没有出现双边滤波常见的 halo。
- JSON 文件 `kodim04_r8_eps1e-03.json` 记录了输入路径与参数，方便撰写实验表格。

#### 6.1.2 `kodim15`（人物）

- `radius=4`、`eps=1e-4` 时脸部细节仍非常清晰；当 `eps` 提升至 `1e-2`，细微噪声被进一步抑制。
- `kodim15_r16_eps1e-03_comparison.png` 显示强平滑效果下的衣服纹理仍可辨认，表明导引滤波在大窗口时仍具有边缘保持能力。

#### 6.1.3 `kodim19`（自然景物）

- 小半径时树叶与道路纹理清晰，大半径时地面纹理被平滑但边界线条仍脆。
- `kodim19_r8_eps1e-04_comparison.png` 与 `kodim19_r16_eps1e-02_comparison.png` 对比可见，`eps` 提高会减少纹理保留，适合噪声较大的场景。

#### 6.1.4 `kodim22`（建筑）

- 建筑边缘在所有参数组合中都保持清楚。
- `eps` 较大时，天空区域几乎纯净，展现导引滤波对大片平坦区域的平滑能力。

### 6.2 细节增强实验

结果位于 `data/results/enhance/`。命名格式：

```
<stem>_alpha<alpha>_r<radius>_eps<eps>.png
<stem>_alpha<alpha>_r<radius>_eps<eps>_comparison.png
<stem>_alpha<alpha>_r<radius>_eps<eps>.json
```

#### 6.2.1 `kodim04` 细节增强

- `alpha=1.0`（`kodim04_alpha1_r8_eps1e-03_comparison.png`）与原图接近，增强效果温和。
- `alpha=1.5`（`kodim04_alpha1.5_r8_eps1e-03_comparison.png`）使衣服皱褶与背景纹理更加明显。
- `alpha=2.0`（`kodim04_alpha2_r8_eps1e-03_comparison.png`）带来最明显的细节提升，皮肤细节被增强到接近 HDR 的对比度，但未出现明显伪影。

#### 6.2.2 `kodim15` 细节增强

- `alpha=1.0` 保留自然质感；
- `alpha=1.5` 时脸部高光与细纹更加突出；
- `alpha=2.0` 显示出明显锐化效果，背景的墙面纹理更加清晰，同时噪声轻微放大，体现了细节增强与噪声放大的权衡关系。

总体而言，实验验证了论文中的公式 `I_enhanced = I + alpha * (I - GF(I))` 的有效性。适当调整 `alpha` 可以控制增强力度而不引入传统高通滤波的伪影。

### 6.3 联合上采样实验

结果位于 `data/results/joint_upsample/`，命名格式：

```
<stem>_scale<scale>_<upsample_method>_r<radius>_eps<eps>_filtered.png
<stem>_scale<scale>_<upsample_method>_r<radius>_eps<eps>_baseline.png
<stem>_scale<scale>_<upsample_method>_r<radius>_eps<eps>_comparison.png
<stem>_scale<scale>_<upsample_method>_r<radius>_eps<eps>.json
```

实验流程：

1. 将高分辨率导引图缩放到 `scale=0.25`，获得低分辨率输入；
2. 使用最近邻插值上采样回原尺寸作为初始输入；
3. 使用高分辨率导引图执行导引滤波，半径为 4，`eps=1e-4`；
4. baseline 使用双线性插值；
5. 生成 `baseline.png`、`filtered.png` 与 `comparison.png`（左右拼接对比）。

#### 6.3.1 `kodim19` 联合上采样

- `kodim19_scale0.25_nearest_r4_eps1e-04_comparison.png` 左侧是双线性上采样，右侧是导引滤波结果。
- 导引滤波结果中，道路边缘与树木轮廓更清晰，无需额外的细节增强操作；
- JSON 文件记录所有参数，便于报告引用。

#### 6.3.2 `kodim22` 联合上采样

- 建筑边缘和天空的过渡更加锐利；
- baseline 图中存在明显的模糊，导引滤波版本恢复了许多高频信息；
- 该实验验证了导引滤波在联合上采样场景的优势，与论文中强调的“边缘一致性”结论一致。

## 7. 结果总结与讨论

1. **边缘保持能力**：在所有平滑实验中，导引滤波表现出高度的边缘保持性，尤其体现在 `radius=16`、`eps=1e-4` 等强平滑设置下仍未出现梯度反转。
2. **参数敏感性**：`radius` 越大，平滑范围越广；`eps` 越大，滤波器越倾向于线性回归，在纹理区域会更快压制高频细节。实际使用时可根据噪声水平选择。
3. **细节增强权衡**：提高 `alpha` 能有效提升细节，但也会相应放大噪声，需要与 `eps` 配合使用；实验显示 `alpha=1.5` 是较为均衡的选择。
4. **联合上采样优势**：导引滤波凭借高分辨率导引图提供的结构信息，在保留边缘的同时避免了普通插值造成的模糊。这一结果与论文 Fig. 10 中的示例一致。
5. **实现效率**：使用 `cv2.boxFilter` 保证盒滤波在 `O(N)` 时间内完成，并且 `numpy.einsum` 使彩色分量解算矢量化，相比逐像素求解大幅加速。

## 8. 误差来源与改进空间

1. **数值稳定性**：极端情况下，局部协方差矩阵可能接近奇异，本实现通过加上 `eps * I` 进行正则；若仍出现问题，可考虑增加更大的正则项或采用更稳定的求逆方法。
2. **边界处理**：当前使用反射边界，若输入图像边界变化很剧烈，可能引入轻微的边缘效应，可考虑切换为常数边界或复制边界。
3. **颜色空间**：实现和论文一样使用 RGB 通道。如需更高质量，可考虑在 YUV 或 Lab 空间中操作，再转换回 RGB。
4. **实验范围**：当前实验覆盖三类典型应用，与论文一致。进一步可扩展到去雾、HDR 压缩等场景，并与其他方法进行定量比较。
5. **性能优化**：对于超高分辨率图，可探索使用积分图或并行化进一步提升速度。

## 9. 复现指南

1. 克隆仓库并安装依赖：
   ```
   uv sync
   ```
2. 将待测试图像放入 `data/input/portrait/` 或 `data/input/landscape/`。
3. 可单独运行某个脚本，例如：
   ```
   uv run python src/demo_smoothing.py --input data/input/landscape/kodim22.png --radius 8 --eps 1e-3
   uv run python src/demo_enhance.py --input data/input/portrait/kodim04.png --alpha 1.5 --radius 8 --eps 1e-3
   uv run python src/demo_joint_upsample.py --lowres data/input/landscape/kodim19.png --guide data/input/landscape/kodim19.png --radius 4 --eps 1e-4
   ```
   （联合上采样脚本会自动创建低分辨率版本，无需手动准备）
4. 批量运行：
   ```
   uv run python scripts/run_suite.py
   ```
5. 查看结果：
   - `data/results/smoothing/`：原图与滤波对比图、参数 JSON；
   - `data/results/enhance/`：原图与增强对比图、参数 JSON；
   - `data/results/joint_upsample/`：baseline 与导引滤波对比图、参数 JSON。
6. 如需调整参数，修改命令行选项后再次运行即可。

## 10. 结论

通过本次复现，我们在 Python 环境下成功还原了导引滤波器的核心算法与论文中的三个典型实验，生成的图像结果与论文描述高度一致。主要成果包括：

1. 完整的 `guided_filter` 函数实现（支持灰度与彩色导引）；
2. 自引导平滑、细节增强、联合上采样脚本与批量运行工具；
3. 系统化的实验结果目录与参数日志；
4. 本报告作为复现文档，概述了实现方法、实验设计与关键观察。

该复现可作为学习导引滤波的入门项目，也可作为进一步研究边缘保留滤波器的基线。未来可尝试结合更复杂的应用（如去雾、图像抠图）或与深度学习模型融合，以探索导引滤波在现代视觉任务中的新可能。

## 11. 附录：结果文件索引（节选）

- `data/results/smoothing/kodim04_r4_eps1e-04_comparison.png`
- `data/results/smoothing/kodim04_r16_eps1e-02_comparison.png`
- `data/results/smoothing/kodim22_r8_eps1e-03_comparison.png`
- `data/results/enhance/kodim04_alpha2_r8_eps1e-03_comparison.png`
- `data/results/enhance/kodim15_alpha1.5_r8_eps1e-03_comparison.png`
- `data/results/joint_upsample/kodim19_scale0.25_nearest_r4_eps1e-04_comparison.png`
- `data/results/joint_upsample/kodim22_scale0.25_nearest_r4_eps1e-04_comparison.png`

以上索引涵盖平滑、细节增强、联合上采样三类实验的代表性输出，可直接用于撰写论文复现报告或制作对比图表。

## 12. 参考文献

1. Kaiming He, Jian Sun, Xiaoou Tang. "Guided Image Filtering." *European Conference on Computer Vision* (ECCV), 2010.
2. Kaiming He, Jian Sun, Xiaoou Tang. "Guided Image Filtering." *IEEE Transactions on Pattern Analysis and Machine Intelligence* (TPAMI), 2013.
3. 本仓库 `paper/eccv10guidedfilter.md` 文件中提供论文全文，供深入阅读和引用。
