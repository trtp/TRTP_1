import json
import re
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from modelscope import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

model_path = "/media/ubuntu/10B4A468B4A451D0/models/InternVL2_5-8B"

# # 数据集路径
# dataset_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_InternVL2-8B-prompt.json"  # 数据集文件路径
# dataset_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_InternVL2_5-8B-prompt.json"
# dataset_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_Llama-3.2-11B-_prompt.json"
# dataset_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_llava-onevision-qwen2-7b-ov-hf-prompt.json"
# dataset_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_llava-v1.6-vicuna-7b-hf-prompt.json"
# dataset_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_MiniCPM-V-2_prompt.json"
# dataset_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_Molmo-7B-D-0924-prompt.json"
# dataset_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_Ovis1.6-Gemma2-9B_prompt.json"
# dataset_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_Qwen2VL7B_prompt.json"
#
# save_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_InternVL2-8B-prompt-InternVL2Evaluation.json"  # 结果保存路径
# save_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_InternVL2_5-8B-prompt-InternVL2Evaluation.json"
# save_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_Llama-3.2-11B-_prompt-InternVL2Evaluation.json"
# save_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_llava-onevision-qwen2-7b-ov-hf-prompt-InternVL2Evaluation.json"
# save_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_llava-v1.6-vicuna-7b-hf-prompt-InternVL2Evaluation.json"
# save_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_MiniCPM-V-2_prompt-InternVL2Evaluation.json"
# save_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_Molmo-7B-D-0924-prompt-InternVL2Evaluation.json"
# save_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_Ovis1.6-Gemma2-9B_prompt-InternVL2Evaluation.json"
# save_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_Qwen2VL7B_prompt-InternVL2Evaluation.json"

# 数据集路径和保存路径列表

dataset_save_pairs = [
#     # InternVL2_5-8B_infer
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_InternVL2-8B-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_InternVL2-8B-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_InternVL2_5-8B-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_InternVL2_5-8B-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_Llama-3.2-11B-_prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_Llama-3.2-11B-_prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_llava-onevision-qwen2-7b-ov-hf-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_llava-onevision-qwen2-7b-ov-hf-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_llava-v1.6-vicuna-7b-hf-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_llava-v1.6-vicuna-7b-hf-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_MiniCPM-V-2_prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_MiniCPM-V-2_prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_Molmo-7B-D-0924-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_Molmo-7B-D-0924-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_Ovis1.6-Gemma2-9B_prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_Ovis1.6-Gemma2-9B_prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2_5-8B_infer_Qwen2VL7B_prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2_5-8B_infer_Qwen2VL7B_prompt-InternVL2Evaluation.json"),
#
# # InternVL2-8B_infer
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_InternVL2-8B-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2-8B_infer_InternVL2-8B-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_InternVL2_5-8B-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2-8B_infer_InternVL2_5-8B-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_Llama-3.2-11B-_prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2-8B_infer_Llama-3.2-11B-_prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_llava-onevision-qwen2-7b-ov-hf-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2-8B_infer_llava-onevision-qwen2-7b-ov-hf-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_llava-v1.6-vicuna-7b-hf-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2-8B_infer_llava-v1.6-vicuna-7b-hf-prompt-InternVL2Evaluation.json"),
#
#     ( "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_MiniCPM-V-2_prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2-8B_infer_MiniCPM-V-2_prompt-InternVL2Evaluation.json"),
#
#     ( "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_Molmo-7B-D-0924-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2-8B_infer_Molmo-7B-D-0924-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_Ovis1.6-Gemma2-9B_prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2-8B_infer_Ovis1.6-Gemma2-9B_prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_Qwen2VL7B_prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/InternVL2-8B_infer_Qwen2VL7B_prompt-InternVL2Evaluation.json"),


# #  Qwen2-VL-7B infer
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/Qwen2-VL-7B_infer_InternVL2-8B-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/Qwen2-VL-7B_infer_InternVL2-8B-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/Qwen2-VL-7B_infer_InternVL2_5-8B-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/Qwen2-VL-7B_infer_InternVL2_5-8B-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/Qwen2-VL-7B_infer_Llama-3.2-11B-_prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/Qwen2-VL-7B_infer_Llama-3.2-11B-_prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/Qwen2-VL-7B_infer_llava-onevision-qwen2-7b-ov-hf-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/Qwen2-VL-7B_infer_llava-onevision-qwen2-7b-ov-hf-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/Qwen2-VL-7B_infer_llava-v1.6-vicuna-7b-hf-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/Qwen2-VL-7B_infer_llava-v1.6-vicuna-7b-hf-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/Qwen2-VL-7B_infer_MiniCPM-V-2_prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/Qwen2-VL-7B_infer_MiniCPM-V-2_prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/Qwen2-VL-7B_infer_Molmo-7B-D-0924-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/Qwen2-VL-7B_infer_Molmo-7B-D-0924-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/Qwen2-VL-7B_infer_Ovis1.6-Gemma2-9B_prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/Qwen2-VL-7B_infer_Ovis1.6-Gemma2-9B_prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/Qwen2-VL-7B_infer_Qwen2VL7B_prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/Qwen2-VL-7B_infer_Qwen2VL7B_prompt-InternVL2Evaluation.json"),

# # llava-onevision-qwen2-7b-ov-hf_infer
# #     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_InternVL2-8B-prompt.json",
# #      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/llava-onevision-qwen2-7b-ov-hf_infer_InternVL2-8B-prompt-InternVL2Evaluation.json"),
# #
# #     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_InternVL2_5-8B-prompt.json",
# #      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/llava-onevision-qwen2-7b-ov-hf_infer_InternVL2_5-8B-prompt-InternVL2Evaluation.json"),
# #
# #     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_Llama-3.2-11B-_prompt.json",
# #      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/llava-onevision-qwen2-7b-ov-hf_infer_Llama-3.2-11B-_prompt-InternVL2Evaluation.json"),
# #
# #     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_llava-onevision-qwen2-7b-ov-hf-prompt.json",
# #      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/llava-onevision-qwen2-7b-ov-hf_infer_llava-onevision-qwen2-7b-ov-hf-prompt-InternVL2Evaluation.json"),
# #
# #     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_llava-v1.6-vicuna-7b-hf-prompt.json",
# #      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/llava-onevision-qwen2-7b-ov-hf_infer_llava-v1.6-vicuna-7b-hf-prompt-InternVL2Evaluation.json"),
# #
# #     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_MiniCPM-V-2_prompt.json",
# #      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/llava-onevision-qwen2-7b-ov-hf_infer_MiniCPM-V-2_prompt-InternVL2Evaluation.json"),
# #
# #     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_Molmo-7B-D-0924-prompt.json",
# #      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/llava-onevision-qwen2-7b-ov-hf_infer_Molmo-7B-D-0924-prompt-InternVL2Evaluation.json"),
# #
# #     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_Ovis1.6-Gemma2-9B_prompt.json",
# #      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/llava-onevision-qwen2-7b-ov-hf_infer_Ovis1.6-Gemma2-9B_prompt-InternVL2Evaluation.json"),
# #
# #     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_Qwen2VL7B_prompt.json",
# #      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/llava-onevision-qwen2-7b-ov-hf_infer_Qwen2VL7B_prompt-InternVL2Evaluation.json"),
#
# # MiniCPM-V-2_6_infer
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_InternVL2-8B-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/MiniCPM-V-2_6_infer_InternVL2-8B-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_InternVL2_5-8B-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/MiniCPM-V-2_6_infer_InternVL2_5-8B-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_Llama-3.2-11B-_prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/MiniCPM-V-2_6_infer_Llama-3.2-11B-_prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_llava-onevision-qwen2-7b-ov-hf-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/MiniCPM-V-2_6_infer_llava-onevision-qwen2-7b-ov-hf-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_llava-v1.6-vicuna-7b-hf-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/MiniCPM-V-2_6_infer_llava-v1.6-vicuna-7b-hf-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_MiniCPM-V-2_prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/MiniCPM-V-2_6_infer_MiniCPM-V-2_prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_Molmo-7B-D-0924-prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/MiniCPM-V-2_6_infer_Molmo-7B-D-0924-prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_Ovis1.6-Gemma2-9B_prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/MiniCPM-V-2_6_infer_Ovis1.6-Gemma2-9B_prompt-InternVL2Evaluation.json"),
#
#     ("/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_Qwen2VL7B_prompt.json",
#      "/home/ubuntu/Desktop/dataset/droidJsonDatset/03Evaluation/MiniCPM-V-2_6_infer_Qwen2VL7B_prompt-InternVL2Evaluation.json")
#


    ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternVL2-8B_infer_llava-onevision-qwen2-7b-ov-hf-prompt_qwen2vlsft.json",
     "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/InternVL2-8B_infer_llava-onevision-qwen2-7b-ov-hf-prompt_qwen2vlsft.json"),

    ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternVL2-8B_infer_Ovis1.6-Gemma2-9B_prompt_qwen2vlsft.json",
     "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/InternVL2-8B_infer_Ovis1.6-Gemma2-9B_prompt_qwen2vlsft.json"),

    ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternVL2-8B_infer_qwen2vlsft.json",
     "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/InternVL2-8B_infer_qwen2vlsft.json"),

    ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternVL2-8B_infer_Qwen2VL7B_prompt_qwen2vlsft.json",
     "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/InternVL2-8B_infer_Qwen2VL7B_prompt_qwen2vlsft.json"),

    ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternVL2_5-8B_infer_InternVL2-8B-prompt_qwen2vlsft.json",
     "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/InternVL2_5-8B_infer_InternVL2-8B-prompt_qwen2vlsft.json"),

    ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternVL2_5-8B_infer_qwen2vlsft.json",
     "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/InternVL2_5-8B_infer_qwen2vlsft.json"),

    ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/MiniCPM-V-2_6_infer_InternVL2_5-8B-prompt_qwen2vlsft.json",
     "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/MiniCPM-V-2_6_infer_InternVL2_5-8B-prompt_qwen2vlsft.json"),

    ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/MiniCPM-V-2_6_infer_Llama-3.2-11B-_prompt_qwen2vlsft.json",
     "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/MiniCPM-V-2_6_infer_Llama-3.2-11B-_prompt_qwen2vlsft.json"),

    ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/MiniCPM-V-2_6_infer_qwen2vlsft.json",
     "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/MiniCPM-V-2_6_infer_qwen2vlsft.json"),

    ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/Qwen2-VL-7B-Instruct_qwen2vlsft.json",
     "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/Qwen2-VL-7B-Instruct_qwen2vlsft.json"),

]

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

# 评分权重（视觉、时序、物理）
alpha, beta, gamma = 0.4, 0.3, 0.3

# 预处理参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_index(bound, fps, max_frame, first_idx=0, num_segments=8):
    """获取关键帧索引"""
    start, end = bound if bound else (-100000, 100000)
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    return np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])


def load_video(video_path, num_segments=8, input_size=448):
    """读取视频并进行预处理"""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame, fps = len(vr) - 1, float(vr.get_avg_fps())
    transform = build_transform(input_size)

    frame_indices = get_index(None, fps, max_frame, first_idx=0, num_segments=num_segments)
    pixel_values_list = []

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = transform(img)
        pixel_values_list.append(img)

    pixel_values = torch.stack(pixel_values_list).to(torch.bfloat16).to(device)
    return pixel_values, num_segments  # 这里 num_segments 代表帧的数量


def evaluate_with_model(video_path, gpt_value):
    """使用 InternVL2-8B 评估视频和 GPT 生成的任务规划"""
    pixel_values, num_patches = load_video(video_path)

    video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(num_patches)])
    question = (
            f"{video_prefix} 根据视频内容，评估以下任务描述是否符合视觉一致性、时序一致性和物理可行性要求（0-100）。视觉一致性确保任务目标出现在场景视频中，并评估其可达性。时序一致性确保任务序列的步骤没有前后依赖冲突问题。物理可行性需要检查任务是否符合物理约束，例如物体是否可移动，目标位置是否允许放置"
            "请严格按照以下格式返回评分： 视觉一致性: <数值>  时序一致性: <数值>  物理可行性: <数值> ："
            + gpt_value
    )

    # 模型推理，假设返回的是一个文本字符串
    outputs = model.chat(
        tokenizer,
        pixel_values,
        question,
        dict(max_new_tokens=32, do_sample=False),
        num_patches_list=[num_patches],
        history=None,
        return_history=False
    )
    return outputs


def parse_scores(output_text):
    """
    解析模型返回的文本，提取视觉、时序和物理的评分
    假设返回格式为：
    "视觉一致性: 90 时序一致性: 85 物理可行性: 80"
    """
    try:
        s_vision = int(re.search(r"视觉一致性:\s*(\d+)", output_text).group(1))
        s_temporal = int(re.search(r"时序一致性:\s*(\d+)", output_text).group(1))
        s_physical = int(re.search(r"物理可行性:\s*(\d+)", output_text).group(1))
    except Exception as e:
        print("解析评分出错：", e)
        s_vision, s_temporal, s_physical = None, None, None
    return s_vision, s_temporal, s_physical


import json
import numpy as np

def evaluate_dataset(dataset_path, save_path):
    """处理单个数据集"""
    # 读取数据集
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    vision_scores, temporal_scores, physical_scores, final_scores = [], [], [], []

    for sample in dataset:
        video_path = sample["videos"][0]
        gpt_value = sample["conversations"][-1]["value"]  # GPT 生成的任务规划

        # 跳过 gpt_value 为 "default" 的样本
        if gpt_value == "default":
            results.append({
                "video": video_path,
                "gpt_value": gpt_value,
                "model_output": None,
                "S_vision": None,
                "S_temporal": None,
                "S_physical": None,
                "final_score": None
            })
            continue

        # 调用评估模型
        output_text = evaluate_with_model(video_path, gpt_value)
        print(f"模型输出 ({dataset_path}):", output_text)

        # 解析评分
        s_vision, s_temporal, s_physical = parse_scores(output_text)

        if s_vision is None or s_temporal is None or s_physical is None:
            final_score = None
        else:
            final_score = alpha * s_vision + beta * s_temporal + gamma * s_physical
            vision_scores.append(s_vision)
            temporal_scores.append(s_temporal)
            physical_scores.append(s_physical)
            final_scores.append(final_score)

        # 保存当前样本的评估结果
        results.append({
            "video": video_path,
            "gpt_value": gpt_value,
            "model_output": output_text,
            "S_vision": s_vision,
            "S_temporal": s_temporal,
            "S_physical": s_physical,
            "final_score": final_score
        })

    # ✅ 仅使用有效评分样本计算均值
    mean_vision = np.mean(vision_scores) if vision_scores else None
    mean_temporal = np.mean(temporal_scores) if temporal_scores else None
    mean_physical = np.mean(physical_scores) if physical_scores else None
    mean_final = np.mean(final_scores) if final_scores else None

    # 插入均值数据到结果文件顶部
    summary = {
        "mean_S_vision": mean_vision,
        "mean_S_temporal": mean_temporal,
        "mean_S_physical": mean_physical,
        "mean_final_score": mean_final
    }

    final_results = [summary] + results

    # 保存评估结果到 JSON 文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"评估完成，结果已保存至 {save_path}")

# 依次处理所有数据集
for dataset_path, save_path in dataset_save_pairs:
    evaluate_dataset(dataset_path, save_path)
#
# # 读取数据集
# with open(dataset_path, "r", encoding="utf-8") as f:
#     dataset = json.load(f)
#
# results = []
# vision_scores, temporal_scores, physical_scores, final_scores = [], [], [], []
#
# for sample in dataset:
#     video_path = sample["videos"][0]
#     gpt_value = sample["conversations"][-1]["value"]  # GPT 生成的任务规划
#
#     # 调用评估模型
#     output_text = evaluate_with_model(video_path, gpt_value)
#     print("模型输出:", output_text)
#
#     # 解析评分
#     s_vision, s_temporal, s_physical = parse_scores(output_text)
#     if s_vision is None or s_temporal is None or s_physical is None:
#         final_score = None
#     else:
#         # 计算最终评分
#         final_score = alpha * s_vision + beta * s_temporal + gamma * s_physical
#
#         # 存储各指标分数
#         vision_scores.append(s_vision)
#         temporal_scores.append(s_temporal)
#         physical_scores.append(s_physical)
#         final_scores.append(final_score)
#
#     # 保存当前样本的评估结果
#     results.append({
#         "video": video_path,
#         "gpt_value": gpt_value,
#         "model_output": output_text,
#         "S_vision": s_vision,
#         "S_temporal": s_temporal,
#         "S_physical": s_physical,
#         "final_score": final_score
#     })
#
# # 计算均值
# mean_vision = np.mean(vision_scores) if vision_scores else None
# mean_temporal = np.mean(temporal_scores) if temporal_scores else None
# mean_physical = np.mean(physical_scores) if physical_scores else None
# mean_final = np.mean(final_scores) if final_scores else None
#
# # 插入均值数据到结果文件顶部
# summary = {
#     "mean_S_vision": mean_vision,
#     "mean_S_temporal": mean_temporal,
#     "mean_S_physical": mean_physical,
#     "mean_final_score": mean_final
# }
#
# final_results = [summary] + results
#
# # 保存评估结果到 JSON 文件
# with open(save_path, "w", encoding="utf-8") as f:
#     json.dump(final_results, f, indent=4, ensure_ascii=False)
#
# print(f"评估完成，结果已保存至 {save_path}")