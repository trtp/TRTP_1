import os
import cv2


def extract_frame_from_videos(source_dir, target_dir, extensions, frame_time=1):
    """
    从每个视频文件中截取一帧图片，并保存到目标目录。

    :param source_dir: 包含视频文件的源目录。
    :param target_dir: 保存截取图片的目标目录。
    :param extensions: 视频文件的扩展名列表（如[".mp4", ".avi"]）。
    :param frame_time: 截取时间（单位为秒，默认为第1秒）。
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    video_counter = 0  # 视频计数器

    for root, _, files in os.walk(source_dir):
        for file in files:
            # 检查文件扩展名是否为指定的视频格式
            if any(file.lower().endswith(ext) for ext in extensions):
                video_path = os.path.join(root, file)
                video_name = os.path.splitext(file)[0]

                # 打开视频文件
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"无法打开视频文件: {video_path}")
                    continue

                # 设置视频读取的位置（按时间）
                fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
                frame_number = int(fps * frame_time)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                # 读取帧
                success, frame = cap.read()
                if success:
                    # 保存帧为图片
                    output_path = os.path.join(target_dir, f"{video_name}.jpg")
                    cv2.imwrite(output_path, frame)
                    print(f"截取图片: {output_path}")
                else:
                    print(f"无法从 {video_path} 截取帧")

                # 释放视频对象
                cap.release()
                video_counter += 1

    print(f"总共处理了 {video_counter} 个视频文件。")


# 使用示例
source_directory = "/home/ubuntu/Desktop/dataset/droid"  # 替换为包含视频文件的源目录路径
target_directory = "/home/ubuntu/Desktop/dataset/droidCutImage"  # 替换为保存截取图片的目标目录路径
video_extensions = [".mp4", ".avi", ".mov", ".mkv"]  # 支持的视频格式
frame_time = 1  # 从第1秒截取

extract_frame_from_videos(source_directory, target_directory, video_extensions, frame_time)