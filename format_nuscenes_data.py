import open3d as o3d
import numpy as np

PCD_PATH = "../v1.0-mini/samples/RADAR_FRONT/n008-2018-08-01-15-16-36-0400__RADAR_FRONT__1533151603555991.pcd"
print("Testing IO for point cloud ...")
pcd = o3d.io.read_point_cloud(PCD_PATH, format='pcd')
print(pcd)
print(pcd.points)

# arr = np.array(pcd)
arr = np.asarray(pcd.points)

print("arr:", arr)
