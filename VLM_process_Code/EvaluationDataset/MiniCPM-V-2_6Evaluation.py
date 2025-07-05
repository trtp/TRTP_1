import json
import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
from decord import VideoReader, cpu  # pip install decord

# Parameter settings
dataset_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/qwenvl_2b_qwenvl_2b_updated_dataset.json"  # Dataset file path
save_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/qwenvl_2b_qwenvl_2b_updated_dataset_EvaluationResult.json"  # Path to save results
model_path = "/media/ubuntu/10B4A468B4A451D0/models/MiniCPM-V-2_6"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                  attn_implementation='sdpa', torch_dtype=torch.bfloat16).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Scoring weights
alpha, beta, gamma = 0.4, 0.3, 0.3

MAX_NUM_FRAMES = 64  # Maximum number of frames to sample


def encode_video(video_path):
    """Extract keyframes from a video."""

    def uniform_sample(l, n):
        """Uniformly samples n elements from a list l."""
        if len(l) <= n:
            return l
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # Sample 1 frame per second
    frame_idx = [i for i in range(0, len(vr), sample_fps)]

    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)

    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]

    print(f'Video {video_path} sampled {len(frames)} frames')
    return frames


def evaluate_with_model(video_path, gpt_value, eval_type):
    """Evaluate task consistency using MiniCPM-V-2.6."""
    frames = encode_video(video_path)

    # The prompt for the model
    question = f"Please evaluate if the task description conforms to {eval_type}, and provide a score from 0 to 1: " + gpt_value
    msgs = [{'role': 'user', 'content': frames + [question]}]

    # Set inference parameters
    params = {"use_image_id": False, "max_slice_nums": 2}

    # Inference
    answer = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, **params)

    # Parse the score
    try:
        score = float(answer.strip())
        score = max(0.0, min(1.0, score))  # Clamp the value between 0 and 1
    except (ValueError, TypeError):
        score = 0.5  # Default value on parsing failure

    return score


# Load the dataset
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

results = []
for sample in dataset:
    video_path = sample["videos"][0]
    gpt_value = sample["conversations"][-1]["value"]

    s_vision = evaluate_with_model(video_path, gpt_value, "visual consistency")
    s_temporal = evaluate_with_model(video_path, gpt_value, "temporal consistency")
    s_physical = evaluate_with_model(video_path, gpt_value, "physical feasibility")

    # Calculate the final score
    final_score = alpha * s_vision + beta * s_temporal + gamma * s_physical

    results.append({
        "video": video_path,
        "gpt_value": gpt_value,
        "S_vision": s_vision,
        "S_temporal": s_temporal,
        "S_physical": s_physical,
        "final_score": final_score
    })

# Save results
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Evaluation complete. Results have been saved to {save_path}")