from io import BytesIO
import h5py
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# 打开 HDF5 文件
file_path = "/home/ubuntu/Desktop/dataset/h5_ur_1rgb/bread_in_basket_1/success_episodes/train/1014_140258/data/trajectory.hdf5"
output_video_path = "../../output_video.mp4"

with h5py.File(file_path, "r") as hdf_file:
    # 访问 RGB 图像数据
    rgb_images = hdf_file["observations"]["rgb_images"]["camera_top"]
    frame_count = len(rgb_images)

    # 解码第一帧以获取尺寸
    first_image = Image.open(BytesIO(rgb_images[0]))
    frame_width, frame_height = first_image.size

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用 H.264 编码器需要安装合适的编解码库
    fps = 30  # 假设帧率为 10，可根据需要调整
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not video_writer.isOpened():
        raise RuntimeError("视频写入器无法打开，请检查编码器参数和输出路径！")

    # 解码并写入每一帧
    for i in tqdm(range(frame_count), desc="Processing frames"):
        # 将字节流解码为图像
        image = Image.open(BytesIO(rgb_images[i]))
        image_np = np.array(image)  # 转为 NumPy 数组
        frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # 转换为 BGR 格式

        # 检查帧尺寸是否正确
        if frame_bgr.shape[:2] != (frame_height, frame_width):
            raise ValueError(f"第 {i} 帧的尺寸不匹配: {frame_bgr.shape[:2]}")

        video_writer.write(frame_bgr)

    # 释放视频写入器
    video_writer.release()
    print(f"视频保存成功: {output_video_path}")

# 使用 OpenCV 检查生成的视频
cap = cv2.VideoCapture(output_video_path)
if not cap.isOpened():
    raise RuntimeError("生成的视频无法读取，请检查编码器或文件完整性！")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"生成的视频帧数: {frame_count}")
cap.release()