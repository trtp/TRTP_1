import json
import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
from decord import VideoReader, cpu  # pip install decord

# 参数设置
dataset_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/qwenvl_2b_qwenvl_2b_updated_dataset.json"  # 数据集文件路径
save_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/qwenvl_2b_qwenvl_2b_updated_dataset_EvaluationResult.json"  # 结果保存路径
model_path = "/media/ubuntu/10B4A468B4A451D0/models/MiniCPM-V-2_6"

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                  attn_implementation='sdpa', torch_dtype=torch.bfloat16).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 评分权重
alpha, beta, gamma = 0.4, 0.3, 0.3

MAX_NUM_FRAMES = 64  # 取样最大帧数


def encode_video(video_path):
    """从视频中提取关键帧"""

    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # 取 1 秒 1 帧
    frame_idx = [i for i in range(0, len(vr), sample_fps)]

    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)

    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]

    print(f'视频 {video_path} 采样 {len(frames)} 帧')
    return frames


def evaluate_with_model(video_path, gpt_value, eval_type):
    """使用 MiniCPM-V-2_6 评估任务一致性"""
    frames = encode_video(video_path)

    question = f"请评估任务描述是否符合{eval_type}，并用0-1评分：" + gpt_value
    msgs = [{'role': 'user', 'content': frames + [question]}]

    # 设置推理参数
    params = {"use_image_id": False, "max_slice_nums": 2}

    # 推理
    answer = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, **params)

    # 解析得分
    try:
        score = float(answer.strip())
        score = max(0.0, min(1.0, score))  # 限制在 0-1 之间
    except ValueError:
        score = 0.5  # 解析失败默认值

    return score


# 读取数据集
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

results = []
for sample in dataset:
    video_path = sample["videos"][0]
    gpt_value = sample["conversations"][-1]["value"]

    s_vision = evaluate_with_model(video_path, gpt_value, "视觉一致性")
    s_temporal = evaluate_with_model(video_path, gpt_value, "时序一致性")
    s_physical = evaluate_with_model(video_path, gpt_value, "物理可行性")

    # 计算最终得分
    final_score = alpha * s_vision + beta * s_temporal + gamma * s_physical

    results.append({
        "video": video_path,
        "gpt_value": gpt_value,
        "S_vision": s_vision,
        "S_temporal": s_temporal,
        "S_physical": s_physical,
        "final_score": final_score
    })

# 保存结果
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"评估完成，结果已保存至 {save_path}")