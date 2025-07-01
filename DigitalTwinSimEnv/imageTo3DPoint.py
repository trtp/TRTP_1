
#-*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
import open3d as o3d
import os

# 1. 加载深度图（16位TIFF图像）
depth_image_path = "vis_depth/1_depth.tiff"  # 修改为生成的深度图路径
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

# 2. 设置相机内参（假设你有相机的内参，如果没有，可能需要进行相机标定）
fx, fy = 1800, 1800  # 假设焦距为800，实际中根据你的相机进行调整
cx, cy = depth_image.shape[1] / 2, depth_image.shape[0] / 2  # 主点设为图像中心

# 3. 从深度图生成点云
height, width = depth_image.shape
points = []

# 深度值是16位的，单位是毫米，转换为米
depth_image = depth_image.astype(np.float32) / 100.0  # 转换为米

for v in range(height):
    for u in range(width):
        Z = depth_image[v, u]  # 深度值
        if Z > 0:  # 排除无效深度值
            X = (u - cx) * Z / fx  # 计算X坐标
            Y = (v - cy) * Z / fy  # 计算Y坐标
            points.append([X, Y, Z])  # 添加到点云列表

# 转换为Open3D的点云对象
points = np.array(points)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 4. 可视化点云
o3d.visualization.draw_geometries([pcd])

# 5. 可选：保存点云为PLY格式
ply_path = "output_point_cloud.ply"
o3d.io.write_point_cloud(ply_path, pcd)
print(f"Point cloud saved to {ply_path}")

