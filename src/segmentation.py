import os
import numpy as np
import networkx as nx
from typing import Dict, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt

def _grid_index(x, y, origin_xy, grid):
    x0, y0 = origin_xy
    ix = np.clip(((x - x0) / grid).astype(int), 0, None)
    iy = np.clip(((y - y0) / grid).astype(int), 0, None)
    return ix, iy

def _region_adjacency_and_edge_strength(labels: np.ndarray, ridge: np.ndarray) -> Tuple[nx.Graph, dict]:
    """
    构建 RAG：节点为分水岭区域ID，边权为边界处屋脊强度（均值）。
    """
    H, W = labels.shape
    G = nx.Graph()
    boundary_samples = defaultdict(list)

    # 扫描4邻接，统计不同label接触处的ridge值
    for y in range(H - 1):
        for x in range(W - 1):
            a = labels[y, x]
            b = labels[y, x + 1]
            c = labels[y + 1, x]
            if a != b and a != 0 and b != 0:
                e = tuple(sorted((int(a), int(b))))
                boundary_samples[e].append(ridge[y, x + 1])
            if a != c and a != 0 and c != 0:
                e = tuple(sorted((int(a), int(c))))
                boundary_samples[e].append(ridge[y + 1, x])

    # 建图
    for e, vals in boundary_samples.items():
        i, j = e
        G.add_node(i); G.add_node(j)
        w = float(np.mean(vals))  # 屋脊强度（边界平均梯度）
        G.add_edge(i, j, weight=w)

    return G, boundary_samples

def _rag_prune_and_merge(G: nx.Graph, ridge_thresh: float, merge_thresh: float):
    """
    基于阈值裁剪/合并：
    - 强边（>= ridge_thresh） → 保留为分割边（不相连）
    - 弱边（<= merge_thresh） → 认为同一建筑，合并
    - 介于两者之间的边 → 保留连接（后续连通分量自然归并）
    合并通过“删除强边”+“记录弱边的合并意图”来实现。
    """
    # 删除强边（切断图）
    to_remove = []
    for u, v, d in G.edges(data=True):
        if d.get("weight", 0.0) >= ridge_thresh:
            to_remove.append((u, v))
    G.remove_edges_from(to_remove)

    # 对极弱边，标记为“强制合并”（这里用连通性自然实现，无需额外并查集）
    # 实际上删除强边后，弱边只要存在就维持连通，已达成“合并”效果。
    # 若想更严格，可在弱边上降低权重，但这里不再多做处理。

def _labels_to_instances_for_roof(points, classes, labels, origin_xy, grid_size):
    """将 roof 点（XY落在网格）映射到对应的分水岭区域ID，作为初步实例ID；非roof先置-1。"""
    inst = np.full(len(points), -1, dtype=np.int64)
    roof_mask = classes == 1
    H, W = labels.shape
    ix, iy = _grid_index(points[roof_mask, 0], points[roof_mask, 1], origin_xy, grid_size)
    # 限定到网格范围
    ix = np.clip(ix, 0, W - 1)
    iy = np.clip(iy, 0, H - 1)
    inst_roof = labels[iy, ix]
    inst[roof_mask] = inst_roof
    return inst

def _reindex_instances(instance_ids: np.ndarray):
    """将实例ID压缩到 1..K，未分配(-1或0)保留为0。"""
    uniq = np.unique(instance_ids)
    uniq = uniq[(uniq > 0)]
    remap = {old: i for i, old in enumerate(uniq, start=1)}
    out = np.zeros_like(instance_ids, dtype=np.int64)
    for old, new in remap.items():
        out[instance_ids == old] = new
    return out

def run(preprocessed_file: str, ridge_thresh: float = 0.25, merge_thresh: float = 0.15):
    data = np.load(preprocessed_file, allow_pickle=True).item()
    points = data["points"]
    classes = data["classes"]
    grid_size = float(data["grid_size"])
    dsm = data["dsm"]
    origin_xy = tuple(data["origin_xy"])
    ws_labels = data["ws_labels"]
    ridge_strength = data["ridge_strength"]

    # 构建RAG图
    G, boundary_samples = _region_adjacency_and_edge_strength(ws_labels, ridge_strength)
    print(f"[Seg] 初始区域数: {ws_labels.max()} | 图节点: {G.number_of_nodes()} 边数: {G.number_of_edges()}")

    # 基于屋脊强度的裁剪/合并
    _rag_prune_and_merge(G, ridge_thresh=ridge_thresh, merge_thresh=merge_thresh)

    # 取图的连通分量作为“合并后区域标签”
    comps = list(nx.connected_components(G))
    region_remap = {}
    new_id = 1
    for comp in comps:
        for rid in comp:
            region_remap[rid] = new_id
        new_id += 1

    # 注意：ws_labels中可能有一些区域没有进入图（独立小岛），为它们保留原ID或继续递增
    all_ids = set(np.unique(ws_labels)) - {0}
    missing = all_ids - set(region_remap.keys())
    for rid in sorted(missing):
        region_remap[rid] = new_id
        new_id += 1

    # 生成“合并后标签图”
    H, W = ws_labels.shape
    merged_map = np.zeros_like(ws_labels, dtype=np.int32)
    it = np.nditer(ws_labels, flags=['multi_index'])
    while not it.finished:
        val = int(it[0])
        if val > 0:
            merged_map[it.multi_index] = region_remap[val]
        it.iternext()

    # 将 roof 点映射为实例
    instance_ids = _labels_to_instances_for_roof(points, classes, merged_map, origin_xy, grid_size)
    instance_ids = _reindex_instances(instance_ids)

    # 保存中间结果
    np.save("output/segmentation.npy", {
        "points": points,
        "classes": classes,
        "grid_size": grid_size,
        "origin_xy": origin_xy,
        "merged_map": merged_map,
        "instance_ids": instance_ids
    }, allow_pickle=True)
    print("[Seg] 已保存: output/segmentation.npy")

    # 可视化保存
    os.makedirs("output/debug_visualizations", exist_ok=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.title("Merged Region Map"); plt.imshow(merged_map, cmap="tab20"); plt.axis("off")
    plt.subplot(1,2,2); plt.title("Ridge Strength"); plt.imshow(ridge_strength, cmap="magma"); plt.axis("off")
    plt.tight_layout()
    plt.savefig("output/debug_visualizations/segmentation_overview.png", dpi=200)
    plt.close()
