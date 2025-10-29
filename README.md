## guided-filter-repro

轻量级的导引滤波（Guided Image Filtering, He et al., ECCV 2010 / TPAMI 2013）复现仓库，使用纯 Python 脚本与最小依赖，覆盖平滑、细节增强、联合上采样三个核心实验。论文全文已整理于 `eccv10guidedfilter.md`，便于写作与查阅。

### 环境准备

```bash
# 安装依赖（默认读取 pyproject.toml / uv.lock）
uv sync
```

脚本一律通过 `uv run` 调用：

```bash
uv run python src/demo_smoothing.py --input data/input/example.jpg --radius 8 --eps 1e-3
```

### 数据结构

- `data/input/`：放置原始图像（建议至少包含自然风景、人像各一张，额外准备一对低/高分辨率图用于联合上采样）
- `data/results/<demo>/`：脚本运行后自动生成的结果图与参数记录（JSON）

### 脚本速览

| 脚本 | 说明 | 关键参数 |
| ---- | ---- | -------- |
| `src/demo_smoothing.py` | 自引导平滑/边缘保留，保存滤波结果与对比图 | `--radius`, `--eps` |
| `src/demo_enhance.py` | 细节增强，输出 `I + alpha (I - GF(I))` | `--alpha`, `--radius`, `--eps` |
| `src/demo_joint_upsample.py` | 联合上采样，对比普通插值与导引滤波结果 | `--upsample-method`, `--baseline-method`, `--radius`, `--eps` |

所有脚本会在结果目录内生成 `<stem>_...json` 用于记录输入路径与参数，便于报告复盘。

### 复现实验建议

1. **平滑与边缘保持**：固定输入图像，遍历 `radius ∈ {4, 8, 16}` 与 `eps ∈ {1e-4, 1e-3, 1e-2}`，观察平滑强度与边缘清晰度变化。
2. **细节增强**：在人像或静物图像上测试 `alpha ∈ {1.0, 1.5, 2.0}`，对比增强程度与潜在伪影。
3. **联合上采样**：用低分辨率/高分辨率配对图像，比较普通插值（`--baseline-method`）与导引滤波上采样的边缘锐利度。

将实验结果截图或导出图像，结合 `eccv10guidedfilter.md` 中的公式说明，可快速撰写复现报告。

### 参考

- Kaiming He, Jian Sun, Xiaoou Tang. *Guided Image Filtering*. ECCV 2010; TPAMI 2013.
- 论文全文、原始推导见 `eccv10guidedfilter.md`。
