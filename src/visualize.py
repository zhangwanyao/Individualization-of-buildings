import sys
import numpy as np
import open3d as o3d
import laspy

def read_with_instanceid(path):
    las = laspy.read(path)
    pts = np.vstack((las.x, las.y, las.z)).T
    classes = np.asarray(las.classification, dtype=np.uint8)
    inst = np.asarray(las["InstanceID"], dtype=np.uint32)
    return pts, classes, inst

def colors_from_instance(inst):
    """为不同实例随机着色，0号(未分配)用灰色。"""
    rng = np.random.default_rng(1234)
    uniq = np.unique(inst)
    color_map = {0: np.array([0.6,0.6,0.6])}
    cid = 1
    for u in uniq:
        if u == 0: continue
        color_map[int(u)] = rng.random(3)
        cid += 1
    cols = np.stack([color_map[int(i)] for i in inst], axis=0)
    return cols

def main(path):
    pts, classes, inst = read_with_instanceid(path)
    cols = colors_from_instance(inst)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)

    vis = o3d.visualization.Visualizer()
    vis.create_window("Instances")
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = [1,1,1]
    opt.point_size = 2.0
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python -m src.visualize output/instances_scalar_field.las")
        sys.exit(1)
    main(sys.argv[1])
