import os
import torch
import numpy as np
import json
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def load_model(model_path):
    """
    加载大模型和处理器。
    :param model_path: 模型路径。
    :return: 模型和处理器。
    """
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    return model, tokenizer

def infer_task_planning(model, tokenizer, video_path,prompt, num_segments=8):
    """
    使用大模型推理视频中的任务规划。
    :param model: 加载的大模型。
    :param tokenizer: 模型处理器。
    :param video_path: 视频路径。
    :param num_segments: 分段数。
    :return: 推理结果字符串。
    """
    pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(len(num_patches_list))])
    # question =prompt + video_prefix + '列出视频中的机械手的动作序列，被机械手抓住的物体和动作以及方位要具体'
    question = prompt + video_prefix + '列出视频中的机械手的动作序列'
    response, history = model.chat(tokenizer, pixel_values, question,
                                   generation_config=dict(max_new_tokens=1024, do_sample=True),
                                   num_patches_list=num_patches_list, history=None, return_history=True)

    return response


def update_dataset_with_inference(model, tokenizer, input_json, output_json):
    """
    将大模型推理结果填入数据集的 gpt 字段。
    :param model: 加载的大模型。
    :param tokenizer: 模型处理器。
    :param input_json: 输入 JSON 文件路径。
    :param output_json: 输出 JSON 文件路径。
    """
    # 加载输入数据集
    with open(input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 遍历每一项数据，进行推理
    for item in dataset:
        video_path = item["videos"][0]
        prompt = "这是对视频截图空间场景的描述，给你作为参考" +item["conversations"][0]["value"]
        #prompt = ""
        # 调用推理函数
        print(f"正在处理视频: {video_path}")
        try:
            gpt_result = infer_task_planning(model, tokenizer, video_path,prompt)
            item["conversations"][2]["value"] = gpt_result
            print(f"推理完成: {gpt_result}")
        except Exception as e:
            print(f"推理失败: {video_path}, 错误信息: {e}")

    # 保存更新后的数据集
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"更新后的数据集已保存到: {output_json}")


if __name__ == "__main__":
    # 定义路径
    model_path = '/media/ubuntu/10B4A468B4A451D0/models/InternVL2-8B'

    # 定义 input_json 和 output_json 对应列表
    dataset_pairs = [
        # ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/InternVL2-8B-prompt-output_dataset.json",
        #  "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_InternVL2-8B-prompt.json"),
        #
        # ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/InternVL2_5-8B-prompt-output_dataset.json",
        #  "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_InternVL2_5-8B-prompt.json"),
        #
        # ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Llama-3.2-11B-prompt-output_dataset.json",
        #  "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_Llama-3.2-11B-_prompt.json"),

        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/llava-onevision-qwen2-7b-ov-hf-prompt-output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_llava-onevision-qwen2-7b-ov-hf-prompt.json"),

        # ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/llava-v1.6-vicuna-7b-hf-prompt-output_dataset.json",
        #  "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_llava-v1.6-vicuna-7b-hf-prompt.json"),

        # ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/MiniCPM-V-2_prompt_6output_dataset.json",
        #  "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_MiniCPM-V-2_prompt.json"),

        # ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Molmo-7B-D-0924-prompt-output_dataset.json",
         # "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_Molmo-7B-D-0924-prompt.json"),

        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Ovis1.6-Gemma2-9B_prompt_output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_Ovis1.6-Gemma2-9B_prompt.json"),

        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Qwen2VL7B_prompt_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_Qwen2VL7B_prompt.json")
    ]

    # 加载模型
    model, tokenizer = load_model(model_path)

    # 遍历 dataset_pairs 逐个处理
    for input_json, output_json in dataset_pairs:
        print(f"Processing: {input_json} -> {output_json}")
        update_dataset_with_inference(model, tokenizer, input_json, output_json)
        print(f"Finished processing: {input_json}\n")

# # 使用示例
# if __name__ == "__main__":
#     # 定义路径
#     model_path = '/media/ubuntu/10B4A468B4A451D0/models/InternVL2-8B'
#
#     input_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/02/InternVL2-8B-prompt-output_dataset.json"
#     #  input_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/02/InternVL2_5-8B-prompt-output_dataset.json"
#     # input_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/02/llava-v1.6-vicuna-7b-hf-prompt-output_dataset.json"
#     # input_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/02/MiniCPM-V-2_prompt_6output_dataset.json"
#     # input_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Molmo-7B-D-0924-prompt-output_dataset.json"
#     # input_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Ovis1.6-Gemma2-9B_prompt_output_dataset.json"
#     # input_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Qwen2VL7B_prompt_dataset.json"
#
#     output_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_InternVL2-8B-prompt.json"
#     #  output_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_InternVL2_5-8B-prompt.json"
#     #  output_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_llava-v1.6-vicuna-7b-hf-prompt.json"
#     #  output_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_MiniCPM-V-2_prompt.json"
#     #  output_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_Molmo-7B-D-0924-prompt.json"
#     #  output_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_Ovis1.6-Gemma2-9B_prompt.json"
#     #  output_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_Qwen2VL7B_prompt.json"
#
#     # 加载模型
#     model, tokenizer = load_model(model_path)
#
#     # 更新数据集
#     update_dataset_with_inference(model, tokenizer, input_json, output_json)