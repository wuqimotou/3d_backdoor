import open3d as o3d
import numpy as np
from tqdm import tqdm


txt_path = './backdoorSample/plant_0342.txt'
# 通过numpy读取txt点云
pcd = np.genfromtxt(txt_path, delimiter=",")

pcd_vector = o3d.geometry.PointCloud()
# 加载点坐标
pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])
o3d.visualization.draw_geometries([pcd_vector])
