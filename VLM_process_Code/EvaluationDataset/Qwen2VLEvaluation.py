import json
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

# 参数设置
dataset_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/qwenvl_2b_qwenvl_2b_updated_dataset.json"  # 数据集文件路径
save_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/qwenvl_2b_qwenvl_2b_updated_dataset_EvaluationResult.json"  # 结果保存路径
model_path = "/home/ubuntu/Desktop/Qwen2-VL-2B-Instruct"

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, ignore_mismatched_sizes=True).to(device)
processor = AutoProcessor.from_pretrained(model_path)

# 评分权重
alpha, beta, gamma = 0.4, 0.3, 0.3  # 视觉、时序、物理权重


def evaluate_with_model(video_path, gpt_value, eval_type):
    """使用Qwen2-VL评估视频和GPT生成的任务规划"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"请评估以下任务描述是否符合{eval_type}要求，并用0-1评分：" + gpt_value},
                {"type": "video", "video": video_path},
            ],
        }
    ]

    # 处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], videos=video_inputs, padding=True, return_tensors="pt").to(device)

    # 推理
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=32)
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 提取评分（假设模型返回的是 0-1 的数值）
    try:
        score = float(output_text.strip())
        score = max(0.0, min(1.0, score))  # 限制在 0-1 之间
    except ValueError:
        score = 0.5  # 如果解析失败，给默认值

    return score


# 读取数据集
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

results = []
for sample in dataset:
    video_path = sample["videos"][0]
    gpt_value = sample["conversations"][-1]["value"]  # GPT 生成的任务规划

    s_vision = evaluate_with_model(video_path, gpt_value, "视觉一致性")
    s_temporal = evaluate_with_model(video_path, gpt_value, "时序一致性")
    s_physical = evaluate_with_model(video_path, gpt_value, "物理可行性")

    # 计算最终评分
    final_score = alpha * s_vision + beta * s_temporal + gamma * s_physical

    results.append({
        "video": video_path,
        "gpt_value": gpt_value,
        "S_vision": s_vision,
        "S_temporal": s_temporal,
        "S_physical": s_physical,
        "final_score": final_score
    })

# 保存评估结果
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"评估完成，结果已保存至 {save_path}")