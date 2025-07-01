import json
import os
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from modelscope import AutoModel, AutoTokenizer

# 数据集路径
DATASET_PATH = "/home/ubuntu/Desktop/dataset/droidJsonDatset/InternVL2-8B-prompt-output_dataset.json"
#DATASET_PATH = "/home/ubuntu/Desktop/dataset/droidJsonDatset/InternVL2_5-8B-prompt-output_dataset.json"
DATASET_PATH = "/home/ubuntu/Desktop/dataset/droidJsonDatset/InternVL2-8B-prompt-output_dataset.json"
DATASET_PATH = "/home/ubuntu/Desktop/dataset/droidJsonDatset/InternVL2-8B-prompt-output_dataset.json"
DATASET_PATH = "/home/ubuntu/Desktop/dataset/droidJsonDatset/InternVL2-8B-prompt-output_dataset.json"
DATASET_PATH = "/home/ubuntu/Desktop/dataset/droidJsonDatset/InternVL2-8B-prompt-output_dataset.json"
DATASET_PATH = "/home/ubuntu/Desktop/dataset/droidJsonDatset/InternVL2-8B-prompt-output_dataset.json"
DATASET_PATH = "/home/ubuntu/Desktop/dataset/droidJsonDatset/InternVL2-8B-prompt-output_dataset.json"


RESULTS_PATH = "/home/ubuntu/Desktop/dataset/droidJsonDatsetEvaluation/InternVL2-8B-prompt-output_dataset_Evaluation.json"
#RESULTS_PATH = "/home/ubuntu/Desktop/dataset/droidJsonDatsetEvaluation/InternVL2_5-8B-prompt-output_dataset_Evaluation.json"
RESULTS_PATH = "/home/ubuntu/Desktop/dataset/droidJsonDatsetEvaluation/InternVL2-8B-prompt-output_dataset_Evaluation.json"
RESULTS_PATH = "/home/ubuntu/Desktop/dataset/droidJsonDatsetEvaluation/InternVL2-8B-prompt-output_dataset_Evaluation.json"
RESULTS_PATH = "/home/ubuntu/Desktop/dataset/droidJsonDatsetEvaluation/InternVL2-8B-prompt-output_dataset_Evaluation.json"
RESULTS_PATH = "/home/ubuntu/Desktop/dataset/droidJsonDatsetEvaluation/InternVL2-8B-prompt-output_dataset_Evaluation.json"
RESULTS_PATH = "/home/ubuntu/Desktop/dataset/droidJsonDatsetEvaluation/InternVL2-8B-prompt-output_dataset_Evaluation.json"
RESULTS_PATH = "/home/ubuntu/Desktop/dataset/droidJsonDatsetEvaluation/InternVL2-8B-prompt-output_dataset_Evaluation.json"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_video_frame(video_path, frame_idx=10):
    """读取视频的第10帧"""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    frame_idx = min(frame_idx, total_frames - 1)  # 防止索引超出范围
    frame = vr[frame_idx].asnumpy()
    return Image.fromarray(frame)


def load_image(image, input_size=448):
    """预处理图片，转换为模型输入格式"""
    transform = build_transform(input_size=input_size)
    return transform(image).unsqueeze(0)  # 添加 batch 维度


def evaluate_spatial_relationship(video_path, system_description, model, tokenizer):
    """评估图片和 system 描述的空间关系匹配度"""
    # 读取第 10 帧
    image = load_video_frame(video_path, frame_idx=10)

    # 预处理图片
    pixel_values = load_image(image).to(torch.bfloat16).cuda()

    # 生成输入 prompt
    question = f"<image>\n{system_description}\n根据该图片与输入描述在空间关系上的精确度、完整性和冗余度，并给出评分（0-1）.请严格按照以下格式返回评分：Precision: <数值> Completeness: <数值> Redundancy: <数值>"

    # 推理
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    response = model.chat(tokenizer, pixel_values, question, generation_config)

    # 解析模型输出
    print(f"模型返回结果: {response}")

    # 解析结果中的 Precision, Completeness, Redundancy 评分
    scores = {"Precision": None, "Completeness": None, "Redundancy": None}
    for key in scores.keys():
        try:
            score = float(response.split(key + ":")[1].split()[0])
            scores[key] = round(score, 3)  # 保留 3 位小数
        except Exception as e:
            scores[key] = None  # 解析失败时设为 None

    return scores


# 加载视觉语言模型
path = "/media/ubuntu/10B4A468B4A451D0/models/InternVL2-8B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# 读取数据集
with open(DATASET_PATH, "r") as f:
    dataset = json.load(f)

# 遍历数据集进行评估
for idx, sample in enumerate(dataset):
    system_text = sample["conversations"][0]["value"]
    video_path = sample["videos"][0]

    print(f"评估 {idx + 1}/{len(dataset)}，视频: {video_path}")
    scores = evaluate_spatial_relationship(video_path, system_text, model, tokenizer)

    # 构建当前样本的结果
    current_result = {
        "video": video_path,
        "Precision": scores["Precision"],
        "Completeness": scores["Completeness"],
        "Redundancy": scores["Redundancy"]
    }

    # 如果结果文件存在，则加载已有结果，否则初始化
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            output_data = json.load(f)
        individual_results = output_data.get("individual_results", [])
    else:
        individual_results = []

    # 将当前结果添加到结果列表中
    individual_results.append(current_result)

    # 根据 individual_results 重新计算平均值
    total_scores = {"Precision": 0.0, "Completeness": 0.0, "Redundancy": 0.0}
    valid_count = {"Precision": 0, "Completeness": 0, "Redundancy": 0}
    for res in individual_results:
        for key in total_scores.keys():
            if isinstance(res[key], (int, float)):
                total_scores[key] += res[key]
                valid_count[key] += 1

    average_scores = {}
    for key in total_scores.keys():
        if valid_count[key] > 0:
            average_scores[key] = round(total_scores[key] / valid_count[key], 3)
        else:
            average_scores[key] = "N/A"

    # 构造要保存的字典
    output_data = {
        "individual_results": individual_results,
        "average_scores": average_scores
    }

    # 写回文件，覆盖原文件（这样每次都会保存最新的所有结果）
    with open(RESULTS_PATH, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"当前评估结果已保存到: {RESULTS_PATH}\n")