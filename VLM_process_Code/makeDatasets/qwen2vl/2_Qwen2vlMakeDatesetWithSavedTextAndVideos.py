import os
import json


def get_all_videos(video_folder):
    """
    递归获取视频文件夹中所有视频文件的相对路径。
    :param video_folder: 视频文件夹路径。
    :return: 包含视频文件名和完整路径的字典。
    """
    video_files = {}
    for root, _, files in os.walk(video_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                file_name = os.path.splitext(file)[0]
                file_path = os.path.join(root, file)
                video_files[file_name] = file_path
    return video_files


def create_dataset(txt_folder, video_folder, output_file):
    """
    根据 TXT 文件夹和视频文件夹生成数据集。
    :param txt_folder: 包含场景位置信息的 TXT 文件夹路径。
    :param video_folder: 包含视频文件的文件夹路径。
    :param output_file: 输出数据集的保存路径。
    """
    dataset = []
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    video_files = get_all_videos(video_folder)

    for txt_file in txt_files:
        # 获取文件名（不含扩展名）
        name = os.path.splitext(txt_file)[0]

        # 去掉文件名中的 "_frame" 后缀（如果存在）
        name = name.replace("_frame", "")

        # 找到对应的视频
        video_path = video_files.get(name)
        if not video_path:
            print(f"未找到对应视频: {name}")
            continue

        # 读取 TXT 文件内容
        txt_path = os.path.join(txt_folder, txt_file)
        with open(txt_path, 'r', encoding='utf-8') as f:
            scene_info = f.read().strip()

        # 构造数据项
        data_item = {
            "conversations": [
                {
                    "from": "system",
                    "value": scene_info  # 使用 TXT 中的场景描述作为 system 的值
                },
                {
                    "from": "human",
                    "value": "<video>列出视频中的任务规划"
                },
                {
                    "from": "gpt",
                    "value": "default"
                }
            ],
            "videos": [video_path]
        }
        dataset.append(data_item)
        print(f"已处理: {txt_file} 和 {video_path}")

    # 保存数据集
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"数据集已保存到: {output_file}")


# 使用示例
txt_folder = "/home/ubuntu/Desktop/dataset/droidCutImagePrompt/Qwen2VL7B"  # 替换为 TXT 文件夹路径
video_folder = "/home/ubuntu/Desktop/dataset/droid"  # 替换为视频文件夹路径
output_file = "/home/ubuntu/Desktop/dataset/droidJsonDatset/Qwen2VL7B_prompt_dataset.json"  # 替换为输出数据集路径

create_dataset(txt_folder, video_folder, output_file)