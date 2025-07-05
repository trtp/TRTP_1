#-*- coding: utf-8 -*-
import argparse
import cv2
import numpy as np
import open3d as o3d
import os

# 1. Load the depth map (16-bit TIFF image)
depth_image_path = "vis_depth/1_depth.tiff"  # Change this to the path of your generated depth map
depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

# 2. Set camera intrinsics (assuming you have the camera's intrinsic parameters; if not, you may need to perform camera calibration)
fx, fy = 1800, 1800  # Assuming focal lengths are 1800, adjust according to your camera in practice
cx, cy = depth_image.shape[1] / 2, depth_image.shape[0] / 2  # Set the principal point to the image center

# 3. Generate point cloud from the depth map
height, width = depth_image.shape
points = []

# The depth values are 16-bit, in millimeters, convert them to meters
depth_image = depth_image.astype(np.float32) / 100.0  # Convert to meters

for v in range(height):
    for u in range(width):
        Z = depth_image[v, u]  # Depth value
        if Z > 0:  # Exclude invalid depth values
            X = (u - cx) * Z / fx  # Calculate X coordinate
            Y = (v - cy) * Z / fy  # Calculate Y coordinate
            points.append([X, Y, Z])  # Add to the point cloud list

# Convert to an Open3D PointCloud object
points = np.array(points)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 4. Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

# 5. Optional: Save the point cloud in PLY format
ply_path = "output_point_cloud.ply"
o3d.io.write_point_cloud(ply_path, pcd)
print(f"Point cloud saved to {ply_path}")