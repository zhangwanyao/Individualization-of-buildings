import os
import numpy as np
from typing import Dict
from scipy.ndimage import maximum_filter, gaussian_filter
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.morphology import remove_small_objects, binary_opening, disk
import matplotlib.pyplot as plt

from .io_utils import read_laz_classification, bbox_xy, CLASS_NAMES

def _build_dsm(roof_pts: np.ndarray, grid_size: float):
    """基于 roof 点创建 DSM（网格取最大Z），并返回 DSM、栅格原点与形状。"""
    x0, y0, x1, y1 = bbox_xy(roof_pts)
    nx = int(np.ceil((x1 - x0) / grid_size)) + 1
    ny = int(np.ceil((y1 - y0) / grid_size)) + 1

    dsm = np.full((ny, nx), -np.inf, dtype=np.float32)
    ix = ((roof_pts[:, 0] - x0) / grid_size).astype(int)
    iy = ((roof_pts[:, 1] - y0) / grid_size).astype(int)

    # 取每格最大Z
    for x_ind, y_ind, z in zip(ix, iy, roof_pts[:, 2]):
        if z > dsm[y_ind, x_ind]:
            dsm[y_ind, x_ind] = z

    # 填洞：用最大滤波 + 高斯平滑近邻补洞
    mask = dsm > -np.inf
    if not mask.any():
        raise ValueError("DSM 为空，请检查 roof 点。")

    # 简单洞填：用局部最大替代缺失
    filled = dsm.copy()
    maxf = maximum_filter(np.where(mask, dsm, -1e9), size=3)
    filled[~mask] = maxf[~mask]
    # 平滑
    filled = gaussian_filter(filled, sigma=1.0)

    return filled, (x0, y0), (ny, nx)

def _watershed_roof_regions(dsm: np.ndarray, min_region_points: int):
    """对 -DSM 做分水岭（高处为脊，低处为谷），输出标签图与边界。"""
    # 取梯度作为参考
    grad = sobel(dsm)

    # 寻找种子：在 -grad 或 dsm 局部极大
    # 这里用 dsm 的局部峰作为种子（楼顶高点）
    coordinates = peak_local_max(dsm, footprint=np.ones((5, 5)), labels=None)
    markers = np.zeros(dsm.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coordinates, start=1):
        markers[r, c] = i

    # 分水岭：基于 -dsm（让高处分开）
    labels = watershed(-dsm, markers=markers, mask=np.isfinite(dsm))

    # 去除太小的区域（可能是噪点）
    labels = remove_small_objects(labels, min_size=min_region_points)

    # 屋脊强度近似：梯度（越大越可能是边界）
    ridge_strength = grad / (grad.max() + 1e-6)

    return labels.astype(np.int32), ridge_strength.astype(np.float32)

def _save_debug(dsm, labels, ridge_strength):
    os.makedirs("output/debug_visualizations", exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.title("DSM"); plt.imshow(dsm, cmap="viridis"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title("Watershed Labels"); plt.imshow(labels, cmap="tab20"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title("Ridge Strength"); plt.imshow(ridge_strength, cmap="magma"); plt.axis("off")
    plt.tight_layout()
    plt.savefig("output/debug_visualizations/preprocess_overview.png", dpi=200)
    plt.close()

def run(laz_path: str, grid_size: float = 0.5, min_region_points: int = 100):
    print(f"[Preprocess] 读取: {laz_path}")
    points, classes = read_laz_classification(laz_path)

    # 仅保留 roof + wall + door_window，用于后续贴附（roof 主导）
    roof_mask = classes == 1
    wall_mask = classes == 2
    door_mask = classes == 3

    roof_pts = points[roof_mask]
    if len(roof_pts) < 10:
        raise ValueError("roof 点过少，无法生成 DSM。")

    print(f"[Preprocess] roof点: {len(roof_pts)} | wall点: {wall_mask.sum()} | door_window点: {door_mask.sum()}")

    dsm, origin_xy, shape = _build_dsm(roof_pts, grid_size)
    labels, ridge_strength = _watershed_roof_regions(dsm, min_region_points)

    # 保存中间结果
    np.save("output/preprocessed.npy", {
        "points": points,
        "classes": classes,
        "grid_size": grid_size,
        "dsm": dsm,
        "origin_xy": origin_xy,
        "shape": shape,
        "ws_labels": labels,
        "ridge_strength": ridge_strength
    }, allow_pickle=True)
    print("[Preprocess] 已保存: output/preprocessed.npy")

    _save_debug(dsm, labels, ridge_strength)
    print("[Preprocess] 已保存调试图: output/debug_visualizations/preprocess_overview.png")
