#-*- coding: utf-8 -*-
import open3d as o3d

# 1. 读取点云
pcd = o3d.io.read_point_cloud("output_point_cloud.ply")

# 2. 法线估计（用于重建网格）
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(100)

# 3. Poisson重建
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# （可选）去除低密度区域的伪面（更干净）
import numpy as np
vertices_to_keep = densities > np.quantile(densities, 0.02)
mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])

# 4. 保存为 .obj 或 .stl
o3d.io.write_triangle_mesh("output_model1.obj", mesh)
# o3d.io.write_triangle_mesh("output_model.stl", mesh)
print(" 导出成功：output_model1.obj")
