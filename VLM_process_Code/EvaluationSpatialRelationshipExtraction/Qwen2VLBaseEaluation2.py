import json
import cv2
import torch
import time
import re
import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 1. 加载模型
model_path = "/home/ubuntu/Desktop/Qwen2-VL-2B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    ignore_mismatched_sizes=True,
).to(device)
processor = AutoProcessor.from_pretrained(model_path)

# 2. 读取数据集
dataset_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/InterVL2output_dataset.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

results_file = "evaluation_results.json"

# 如果已有结果文件，则加载之前的结果并累积之前的评分数据
if os.path.exists(results_file):
    with open(results_file, "r", encoding="utf-8") as f:
        existing_data = json.load(f)
    individual_results = existing_data.get("individual_results", [])
    # 重新计算累计评分和有效评分数量
    precision_sum = 0.0
    completeness_sum = 0.0
    redundancy_sum = 0.0
    valid_count = 0
    for r in individual_results:
        if isinstance(r.get("Precision"), (int, float)):
            precision_sum += r["Precision"]
            completeness_sum += r["Completeness"]
            redundancy_sum += r["Redundancy"]
            valid_count += 1
else:
    individual_results = []
    precision_sum = 0.0
    completeness_sum = 0.0
    redundancy_sum = 0.0
    valid_count = 0

# 3. 遍历数据集进行评估
for data in dataset:
    system_value = data["conversations"][0]["value"]  # Ground Truth
    video_path = data["videos"][0]  # 视频路径

    # 读取视频第10帧
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"无法读取 {video_path} 的第10帧")
        continue

    # 保存临时图片
    temp_image_path = "temp_frame.jpg"
    cv2.imwrite(temp_image_path, frame)

    # 构造对话消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请严格评估给定图片和文本描述的空间关系提取能力，并量化以下三个指标："},
                {"type": "text", "text": "1. **精确度（Precision）**: 预测的关系中，正确的占比，避免过多错误信息。"},
                {"type": "text", "text": "2. **完整性（Completeness）**: 是否包含所有关键物体的空间关系描述。"},
                {"type": "text", "text": "3. **冗余度（Redundancy）**: 是否包含过多无关描述，影响理解效率。"},
                {"type": "text", "text": "请基于图片内容和以下 Ground Truth 进行量化评分（0-100）："},
                {"type": "text", "text": f"Ground Truth: {system_value}"},
                {"type": "image", "image": temp_image_path},
            ],
        }
    ]

    # 4. 进行推理
    start_time = time.time()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    end_time = time.time()

    # 解析输出文本，使用正则表达式提取评分
    pattern = r"Precision:\s*([\d\.]+).*Completeness:\s*([\d\.]+).*Redundancy:\s*([\d\.]+)"
    match = re.search(pattern, output_text, re.S)
    if match:
        precision = round(float(match.group(1)), 3)
        completeness = round(float(match.group(2)), 3)
        redundancy = round(float(match.group(3)), 3)
        # 累加有效评分
        precision_sum += precision
        completeness_sum += completeness
        redundancy_sum += redundancy
        valid_count += 1
    else:
        print(f"无法解析 {video_path} 的输出结果")
        precision = None
        completeness = None
        redundancy = None

    # 构建当前样本的结果
    current_result = {
        "video": video_path,
        "ground_truth": system_value,
        "model_output": output_text,
        "inference_time": round(end_time - start_time, 4),
        "Precision": precision,
        "Completeness": completeness,
        "Redundancy": redundancy,
    }

    individual_results.append(current_result)

    # 重新计算平均值
    if valid_count > 0:
        avg_precision = round(precision_sum / valid_count, 3)
        avg_completeness = round(completeness_sum / valid_count, 3)
        avg_redundancy = round(redundancy_sum / valid_count, 3)
    else:
        avg_precision = avg_completeness = avg_redundancy = None

    # 构造待保存的数据字典
    output_data = {
        "individual_results": individual_results,
        "average_scores": {
            "Precision": avg_precision,
            "Completeness": avg_completeness,
            "Redundancy": avg_redundancy
        }
    }

    # 每处理完一个样本就写入保存文件（覆盖写入）
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"处理完 {video_path}，输出: {output_text}")

# 5. 最后打印平均分
print("\n===== 评估平均值 =====")
print(f"Precision: {avg_precision:.2f}" if avg_precision is not None else "Precision: N/A")
print(f"Completeness: {avg_completeness:.2f}" if avg_completeness is not None else "Completeness: N/A")
print(f"Redundancy: {avg_redundancy:.2f}" if avg_redundancy is not None else "Redundancy: N/A")
print("评估完成，结果已保存到 evaluation_results.json")