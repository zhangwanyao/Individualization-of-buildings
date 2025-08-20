# Building Instance Segmentation (传统几何 + 图优化)

本项目对**已完成语义分类**的机载点云进行**建筑单体化**（即：将邻近/粘连但功能独立的建筑拆分成独立实例）。
核心流程：`roof` → DSM → 分水岭初分割 → 基于屋脊/鞍部强度的区域图（RAG）裁剪/合并 → `wall/door_window` 贴附 → 导出 `InstanceID`。

---

##  环境配置

```bash
conda create -n building python=3.9 -y
conda activate building
pip install -r requirements.txt
```


##  数据准备

将 `.laz` 或 `.las` 文件放在 `data/` 文件夹，例如：
```
data/sample.laz
```

分类编码要求：
```
0: others
1: roof
2: wall
3: door_window
4: awning
5: ground
6: vegetation
```

其中 `roof=1`, `wall=2`, `door_window=3` 将用于单体化分割。

---

##  运行流程

### 1) 预处理  
生成 DSM、分水岭初分割、屋脊强度：
```bash
python scripts/run_preprocess.py --input data/sample.laz --grid 0.5 --min_points 100
```

### 2) 分割  
构建 RAG 图，基于屋脊强度裁剪/合并，得到屋面实例：
```bash
python scripts/run_segmentation.py --input output/preprocessed.npy --ridge_thresh 0.25 --merge_thresh 0.15
```

### 3) 后处理  
将 wall/door_window 贴附到最近屋面实例，导出带 `InstanceID` 的 LAS：
```bash
python scripts/run_postprocess.py --input output/segmentation.npy --search_radius 3.0
```

---

## 可视化

简易查看 `InstanceID` 分割结果：
```bash
python -m src.visualize output/instances_scalar_field.las
```

颜色说明：  
- 每个建筑实例随机赋色  
- `InstanceID=0`（未分配）显示为灰色  

---

##  输出文件

- `output/instances_scalar_field.las`  
  带有 `InstanceID` 字段的点云（UInt32 类型）

- `output/debug_visualizations/`  
  DSM、分水岭标签、屋脊强度、合并后区域图等中间结果

---

## 参数说明

- `--grid`：DSM 网格分辨率（米），航飞分辨率好时推荐 0.3~1.0
- `--min_points`：每个候选区域至少包含多少屋顶点（避免过碎区域）
- `--ridge_thresh`：强边阈值（≥该值的边被切断 → 更容易分开）
- `--merge_thresh`：弱边阈值（≤该值的边被合并 → 更容易合并在一起）
- `--search_radius`：立面点附着到屋面实例的邻域半径（米，2~5 常用）

---

## 方法说明

1. **预处理**：提取 roof 点生成 DSM → 分水岭 → 初步屋面区域 → 计算屋脊强度  
2. **分割**：构建 RAG 图 → 强边切断、弱边合并 → 连通分量 = 建筑单体  
3. **后处理**：将 wall/door_window 投影到 XY 平面，附着到最近屋面实例  
4. **导出**：生成带 `InstanceID` 的点云文件，供后续 BIM/统计分析使用  

---

## 注意事项

- 输入点云必须包含 `roof` 点，否则无法生成 DSM。  
- 如果结果**过碎**：调大 `--min_points`、调低 `--ridge_thresh`。  
- 如果结果**分不开**：调高 `--ridge_thresh` 或减小 `--merge_thresh`。  
- 立面贴附的 `--search_radius` 需根据建筑间距调整。  

---
