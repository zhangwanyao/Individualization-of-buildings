import laspy
import numpy as np
import os
from typing import Tuple

# 分类编码：0: awning, 1: door_window, 2: ground, 3: others, 4: roof, 5: vegetation, 6: wall
CLASS_NAMES = [
    "awning",
    "door_window",
    "ground",
    "others",
    "roof",
    "vegetation",
    "wall",
]
CLASS_COLORS = np.array([
    [0.8, 0.8, 0.2],  # awning
    [0.2, 0.2, 0.8],  # door_window
    [0.6, 0.3, 0.1],  # ground
    [0.5, 0.5, 0.5],  # others
    [0.8, 0.2, 0.2],  # roof
    [0.2, 0.8, 0.8],  # vegetation
    [0.2, 0.8, 0.2],  # wall
], dtype=np.float64)

def read_laz_classification(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取LAZ/LAS，返回 Nx3 points 和 N 分类 (uint8)。"""
    try:
        with laspy.open(file_path, laz_backend=laspy.LazBackend.Lazrs) as f:
            las = f.read()
    except:
        try:
            with laspy.open(file_path, laz_backend=laspy.LazBackend.Laszip) as f:
                las = f.read()
        except Exception as e:
            print(f"打开LAZ/LAS失败: {e}")
            print("请安装依赖: pip install lazrs 或 pip install laszip")
            raise
    pts = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    cls = np.asarray(las.classification, dtype=np.uint8)

    valid_mask = cls < len(CLASS_NAMES)
    return pts[valid_mask], cls[valid_mask]

def save_las_with_instance_id(out_path: str,
                              points: np.ndarray,
                              classifications: np.ndarray,
                              instance_ids: np.ndarray):
    """保存 LAS，附加 UInt32 类型的 InstanceID 字段。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = laspy.LasHeader(point_format=3, version="1.2")  # PF3含RGB，可扩展
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.classification = classifications.astype(np.uint8)

    # 附加扩展字段：InstanceID (UInt32)
    if "InstanceID" not in las.point_format.extra_dimension_names:
        extra = laspy.ExtraBytesParams(name="InstanceID", type=np.uint32)
        las.add_extra_dim(extra)
    las["InstanceID"] = instance_ids.astype(np.uint32)

    las.write(out_path)
    print(f"[IO] 保存: {out_path} | 点数: {len(points)}")

def bbox_xy(points: np.ndarray):
    x0, y0 = points[:, 0].min(), points[:, 1].min()
    x1, y1 = points[:, 0].max(), points[:, 1].max()
    return (x0, y0, x1, y1)
