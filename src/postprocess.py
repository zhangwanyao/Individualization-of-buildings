import numpy as np
from scipy.spatial import cKDTree
from .io_utils import save_las_with_instance_id

def _attach_facades(points, classes, instance_ids, search_radius: float = 3.0):
    """
    将 wall/door_window 贴附到最近的 roof 实例（按XY投影的近邻）。
    若找不到邻近roof实例，则保持为0（未分配）。
    """
    roof_mask = (classes == 1) & (instance_ids > 0)
    if roof_mask.sum() == 0:
        print("[Post] 无可用屋面实例，跳过贴附。")
        return instance_ids

    # KDTree 在 XY 平面
    roof_xy = points[roof_mask][:, :2]
    roof_ids = instance_ids[roof_mask]
    tree = cKDTree(roof_xy)

    target_mask = (classes == 2) | (classes == 3)  # wall 或 door_window
    tgt_xy = points[target_mask][:, :2]

    if len(tgt_xy) == 0:
        return instance_ids

    dists, idx = tree.query(tgt_xy, distance_upper_bound=search_radius)
    attach_ids = np.zeros(len(tgt_xy), dtype=np.int64)
    # cKDTree.query若超界会返回 idx=len(data)，dist=inf
    valid = np.isfinite(dists) & (idx < len(roof_xy))
    attach_ids[valid] = roof_ids[idx[valid]]

    out = instance_ids.copy()
    out[target_mask] = np.where(attach_ids > 0, attach_ids, out[target_mask])
    return out

def run(segmentation_file: str, search_radius: float = 3.0):
    data = np.load(segmentation_file, allow_pickle=True).item()
    points = data["points"]
    classes = data["classes"]
    instance_ids = data["instance_ids"]

    print(f"[Post] 初始实例（仅roof）数量: {len(np.unique(instance_ids[instance_ids>0]))}")

    # 立面贴附
    inst2 = _attach_facades(points, classes, instance_ids, search_radius=search_radius)

    print(f"[Post] 贴附后实例数量: {len(np.unique(inst2[inst2>0]))}")

    # 导出带 InstanceID 的LAS
    save_las_with_instance_id("output/instances_scalar_field.las", points, classes, inst2)
