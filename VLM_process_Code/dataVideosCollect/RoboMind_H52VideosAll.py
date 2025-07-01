import os
from io import BytesIO
import h5py
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 根目录路径
root_path = "/home/ubuntu/Desktop/dataset/h5_ur_1rgb"
output_dir = "output_videos"  # 输出视频的目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 视频写入参数
fps = 30  # 帧率
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# 遍历根目录下的所有子目录
for root, dirs, files in os.walk(root_path):
    # 判断当前目录是否为 success_episodes
    if "success_episodes" in root:
        for file_name in files:
            if file_name.endswith(".hdf5"):
                file_path = os.path.join(root, file_name)
                print(f"正在处理文件: {file_path}")

                # 生成唯一的输出视频路径
                relative_path = os.path.relpath(root, root_path)
                unique_name = os.path.join(relative_path, file_name.replace(".hdf5", ".mp4"))
                unique_name = unique_name.replace(os.sep, "_")  # 替换路径分隔符，避免非法文件名
                output_video_path = os.path.join(output_dir, unique_name)

                # 处理 HDF5 文件
                with h5py.File(file_path, "r") as hdf_file:
                    rgb_images = hdf_file["observations"]["rgb_images"]["camera_top"]
                    frame_count = len(rgb_images)

                    # 获取第一帧的尺寸
                    first_image = Image.open(BytesIO(rgb_images[0]))
                    frame_width, frame_height = first_image.size

                    # 初始化视频写入器
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
                    if not video_writer.isOpened():
                        raise RuntimeError(f"视频写入器无法打开: {output_video_path}")

                    # 逐帧解码并写入视频
                    for i in tqdm(range(frame_count), desc=f"Processing {file_name}"):
                        image = Image.open(BytesIO(rgb_images[i]))
                        image_np = np.array(image)
                        frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        video_writer.write(frame_bgr)

                    # 释放当前视频写入器
                    video_writer.release()

                print(f"视频生成成功: {output_video_path}")

print(f"所有视频已生成，保存在目录: {output_dir}")