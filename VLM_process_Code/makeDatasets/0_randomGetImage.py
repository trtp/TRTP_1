import os
import random
import shutil


def random_sample_images(source_folder, output_folder, num_samples):
    """
    从图片文件夹中随机抽取指定数量的图片，并保存到目标文件夹。

    :param source_folder: 源图片文件夹路径。
    :param output_folder: 抽取图片保存的目标文件夹路径。
    :param num_samples: 要抽取的图片数量。
    """
    # 获取图片文件列表
    image_files = [f for f in os.listdir(source_folder) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if len(image_files) < num_samples:
        print(f"图片数量不足，源文件夹仅包含 {len(image_files)} 张图片。")
        return

    # 随机抽取图片
    selected_images = random.sample(image_files, num_samples)

    # 创建目标文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 复制抽取的图片到目标文件夹
    for image in selected_images:
        source_path = os.path.join(source_folder, image)
        target_path = os.path.join(output_folder, image)
        shutil.copy(source_path, target_path)
        print(f"已抽取: {image} -> {target_path}")

    print(f"随机抽取完成，共抽取 {num_samples} 张图片到 {output_folder}")


# 使用示例
source_folder = "/home/ubuntu/Desktop/dataset/droidCutImage"  # 替换为源图片文件夹路径
output_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGetTest"  # 替换为目标文件夹路径
num_samples = 1000  # 替换为要抽取的图片数量

random_sample_images(source_folder, output_folder, num_samples)