#
# import os
# import shutil
#
# def extract_and_rename_videos(source_dir, target_dir, extensions):
#     """
#     批量提取视频文件并重命名（直接移动）。
#
#     :param source_dir: 要搜索视频文件的源目录。
#     :param target_dir: 视频文件提取到的目标目录。
#     :param extensions: 视频文件的扩展名列表（如[".mp4", ".avi"]）。
#     """
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#
#     file_counter = 1  # 用于生成唯一文件名
#
#     # 遍历源目录及其所有子目录
#     for root, _, files in os.walk(source_dir):
#         for file in files:
#             # 检查文件扩展名是否是指定的视频格式
#             if any(file.lower().endswith(ext) for ext in extensions):
#                 source_path = os.path.join(root, file)
#                 # 使用唯一序号重命名文件
#                 new_name = f"video_{file_counter}{os.path.splitext(file)[1]}"
#                 target_path = os.path.join(target_dir, new_name)
#
#                 # 移动文件并重命名
#                 shutil.move(source_path, target_path)
#                 print(f"提取并移动: {source_path} -> {target_path}")
#
#                 file_counter += 1
#
# # 使用示例
# source_directory = "/home/ubuntu/Desktop/dataset/collection-droid"  # 替换为你的源目录路径
# target_directory = "/home/ubuntu/Desktop/dataset/droidMP4"  # 替换为你的目标目录路径
# video_extensions = [".mp4", ".avi", ".mov", ".mkv"]  # 添加需要支持的视频格式
#
# extract_and_rename_videos(source_directory, target_directory, video_extensions)


import os
import shutil


def extract_videos_in_batches(source_dir, target_dir, extensions, batch_size):
    """
    将文件夹中的视频每 batch_size 个提取到一个子文件夹中（直接移动）。

    :param source_dir: 要搜索视频文件的源目录。
    :param target_dir: 视频文件提取到的目标目录。
    :param extensions: 视频文件的扩展名列表（如[".mp4", ".avi"]）。
    :param batch_size: 每个目标子目录包含的最大视频文件数。
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_counter = 0  # 全局计数器，用于计算视频文件总数
    batch_counter = 1  # 批次计数器，用于创建新的子文件夹
    current_batch_dir = os.path.join(target_dir, f"batch_{batch_counter}")
    os.makedirs(current_batch_dir, exist_ok=True)

    # 遍历源目录及其所有子目录
    for root, _, files in os.walk(source_dir):
        for file in files:
            # 检查文件扩展名是否是指定的视频格式
            if any(file.lower().endswith(ext) for ext in extensions):
                source_path = os.path.join(root, file)

                # 如果当前批次文件数量达到上限，创建新的子文件夹
                if file_counter % batch_size == 0 and file_counter > 0:
                    batch_counter += 1
                    current_batch_dir = os.path.join(target_dir, f"batch_{batch_counter}")
                    os.makedirs(current_batch_dir, exist_ok=True)

                # 在当前批次文件夹中生成文件路径
                new_name = f"video_{file_counter + 1}{os.path.splitext(file)[1]}"
                target_path = os.path.join(current_batch_dir, new_name)

                # 移动文件并重命名
                shutil.move(source_path, target_path)
                print(f"提取并移动: {source_path} -> {target_path}")

                file_counter += 1


# 使用示例
source_directory = "/home/ubuntu/Desktop/dataset/droidMP4"  # 替换为你的源目录路径
target_directory = "/home/ubuntu/Desktop/dataset/droid"  # 替换为你的目标目录路径
video_extensions = [".mp4", ".avi", ".mov", ".mkv"]  # 添加需要支持的视频格式
batch_size = 1000  # 每个子文件夹中最多的视频数量

extract_videos_in_batches(source_directory, target_directory, video_extensions, batch_size)
