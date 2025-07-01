# -*- coding: utf-8 -*-
import h5py
import numpy as np
import cv2
import os

# 配置路径和参数
h5_file_path = "E:/code/DataSets/all_play_data_diverse/all_play_data_diverse.h5"
output_video_dir = "videos_all_play_data_diverse"
prompt_file_path = os.path.join(output_video_dir, "prompts.txt")
os.makedirs(output_video_dir, exist_ok=True)  # 创建存储视频的文件夹

# 打开 h5 文件
with h5py.File(h5_file_path, "r") as F:
    # 提取数据
    data = {key: np.array(F[key]) for key in F.keys()}

    # 打开文本文件用于写入
    with open(prompt_file_path, "w") as prompt_file:
        # 遍历终止符，分割技能序列
        start_idx = 0
        skill_count = 0
        for i, is_terminal in enumerate(data['terminals']):
            if is_terminal:
                # 提取当前技能的图像序列
                sequence_images = data['front_cam_ob'][start_idx:i + 1]
                prompt = data['prompts'][skill_count]  # 提取对应的 prompt

                # 设置输出视频路径
                video_name = f"skill_{skill_count}.mp4"
                video_path = os.path.join(output_video_dir, video_name)

                # 保存为视频
                height, width, _ = sequence_images[0].shape
                fps = 10  # 视频帧率
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                for frame in sequence_images:
                    bgr_frame = frame[..., ::-1]  # 将 RGB 转为 BGR（cv2 格式）
                    video_writer.write(bgr_frame)

                video_writer.release()
                print(f"Saved video: {video_path}")

                # 写入 Prompt 与视频对应关系
                prompt_file.write(f"{video_name}: {prompt}\n")
                print(f"Saved prompt: {prompt}")

                # 更新索引和计数
                start_idx = i + 1
                skill_count += 1

print(f"All prompts saved to {prompt_file_path}")
