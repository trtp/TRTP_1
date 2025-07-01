# -*- coding: utf-8 -*-
import os
import cv2

# 按自然序号排序图片文件
def natural_sort(images):
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(images, key=alphanum_key)

def create_video_from_images(image_folder, output_path, fps=30):
    """将指定文件夹中的图片合成一个视频"""
    # 按自然顺序排序图片
    images = natural_sort([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])
    if not images:
        print(f"未找到任何图片: {image_folder}")
        return False

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # 定义视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"无法读取图片: {image_path}")
            continue
        video.write(frame)

    video.release()
    print(f"视频已保存至: {output_path}")
    return True

def process_scripted_raw(base_dir, output_dir, fps=30):
    """处理scripted_raw目录，查找所有images0文件夹并生成视频"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name == "images6":
                image_folder = os.path.join(root, dir_name)
                relative_path = os.path.relpath(root, base_dir)
                output_video_path = os.path.join(output_dir, f"{relative_path.replace(os.sep, '_')}.mp4")
                os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

                print(f"正在处理文件夹: {image_folder}")
                create_video_from_images(image_folder, output_video_path, fps=fps)

if __name__ == "__main__":
    base_dir = "E:/code/DataSets/scripted_raw"  # 根目录
    output_dir = "E:/code/DataSets/scripted_videos4"  # 输出视频保存目录
    fps = 10  # 视频帧率

    process_scripted_raw(base_dir, output_dir, fps=fps)