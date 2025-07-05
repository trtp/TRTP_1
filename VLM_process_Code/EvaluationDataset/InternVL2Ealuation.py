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

dataset_save_pairs = [
    # (Paths are left as they are, as they are specific to your system)
    (
    "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternVL2-8B_infer_llava-onevision-qwen2-7b-ov-hf-prompt_qwen2vlsft.json",
    "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/InternVL2-8B_infer_llava-onevision-qwen2-7b-ov-hf-prompt_qwen2vlsft.json"),
    (
    "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternVL2-8B_infer_Ovis1.6-Gemma2-9B_prompt_qwen2vlsft.json",
    "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/InternVL2-8B_infer_Ovis1.6-Gemma2-9B_prompt_qwen2vlsft.json"),
    ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternVL2-8B_infer_qwen2vlsft.json",
     "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/InternVL2-8B_infer_qwen2vlsft.json"),
    ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternVL2-8B_infer_Qwen2VL7B_prompt_qwen2vlsft.json",
     "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/InternVL2-8B_infer_Qwen2VL7B_prompt_qwen2vlsft.json"),
    ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternVL2_5-8B_infer_InternVL2-8B-prompt_qwen2vlsft.json",
     "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/InternVL2_5-8B_infer_InternVL2-8B-prompt_qwen2vlsft.json"),
    ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternVL2_5-8B_infer_qwen2vlsft.json",
     "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/InternVL2_5-8B_infer_qwen2vlsft.json"),
    (
    "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/MiniCPM-V-2_6_infer_InternVL2_5-8B-prompt_qwen2vlsft.json",
    "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/MiniCPM-V-2_6_infer_InternVL2_5-8B-prompt_qwen2vlsft.json"),
    (
    "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/MiniCPM-V-2_6_infer_Llama-3.2-11B-_prompt_qwen2vlsft.json",
    "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/MiniCPM-V-2_6_infer_Llama-3.2-11B-_prompt_qwen2vlsft.json"),
    ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/MiniCPM-V-2_6_infer_qwen2vlsft.json",
     "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/MiniCPM-V-2_6_infer_qwen2vlsft.json"),
    ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/Qwen2-VL-7B-Instruct_qwen2vlsft.json",
     "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output/InternEvaluation/Qwen2-VL-7B-Instruct_qwen2vlsft.json"),
]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

# Scoring weights (visual, temporal, physical)
alpha, beta, gamma = 0.4, 0.3, 0.3

# Preprocessing parameters
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
    """Get keyframe indices"""
    start, end = bound if bound else (-100000, 100000)
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    return np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])


def load_video(video_path, num_segments=8, input_size=448):
    """Read video and preprocess it"""
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
    return pixel_values, num_segments  # Here num_segments represents the number of frames


def evaluate_with_model(video_path, gpt_value):
    """Evaluate the video and the task plan generated by GPT using InternVL2.5-8B"""
    pixel_values, num_patches = load_video(video_path)

    video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(num_patches)])
    question = (
            f"{video_prefix}Based on the video content, evaluate whether the following task description meets the "
            "requirements for visual consistency, temporal consistency, and physical feasibility (on a scale of 0-100). "
            "Visual consistency ensures that the task's target objects appear in the scene video and evaluates their reachability. "
            "Temporal consistency ensures that there are no dependency conflicts in the sequence of task steps. "
            "Physical feasibility involves checking if the task adheres to physical constraints, such as whether objects "
            "are movable and if the target locations are valid for placement. "
            "Please strictly return the scores in the following format: "
            "Visual consistency: <score>  Temporal consistency: <score>  Physical feasibility: <score> : "
            + gpt_value
    )

    # Model inference, assuming it returns a text string
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
    Parse the text returned by the model to extract visual, temporal, and physical scores.
    Assumes the returned format is:
    "Visual consistency: 90  Temporal consistency: 85  Physical feasibility: 80"
    """
    try:
        s_vision = int(re.search(r"Visual consistency:\s*(\d+)", output_text, re.IGNORECASE).group(1))
        s_temporal = int(re.search(r"Temporal consistency:\s*(\d+)", output_text, re.IGNORECASE).group(1))
        s_physical = int(re.search(r"Physical feasibility:\s*(\d+)", output_text, re.IGNORECASE).group(1))
    except Exception as e:
        print("Error parsing scores:", e)
        s_vision, s_temporal, s_physical = None, None, None
    return s_vision, s_temporal, s_physical


def evaluate_dataset(dataset_path, save_path):
    """Process a single dataset"""
    # Load the dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    vision_scores, temporal_scores, physical_scores, final_scores = [], [], [], []

    for sample in dataset:
        video_path = sample["videos"][0]
        gpt_value = sample["conversations"][-1]["value"]  # Task plan generated by GPT

        # Skip samples where gpt_value is "default"
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

        # Call the evaluation model
        output_text = evaluate_with_model(video_path, gpt_value)
        print(f"Model output ({dataset_path}):", output_text)

        # Parse scores
        s_vision, s_temporal, s_physical = parse_scores(output_text)

        if s_vision is None or s_temporal is None or s_physical is None:
            final_score = None
        else:
            final_score = alpha * s_vision + beta * s_temporal + gamma * s_physical
            vision_scores.append(s_vision)
            temporal_scores.append(s_temporal)
            physical_scores.append(s_physical)
            final_scores.append(final_score)

        # Save the evaluation results for the current sample
        results.append({
            "video": video_path,
            "gpt_value": gpt_value,
            "model_output": output_text,
            "S_vision": s_vision,
            "S_temporal": s_temporal,
            "S_physical": s_physical,
            "final_score": final_score
        })

    #  Calculate the mean using only valid scored samples
    mean_vision = np.mean(vision_scores) if vision_scores else None
    mean_temporal = np.mean(temporal_scores) if temporal_scores else None
    mean_physical = np.mean(physical_scores) if physical_scores else None
    mean_final = np.mean(final_scores) if final_scores else None

    # Insert the summary statistics at the top of the results file
    summary = {
        "mean_S_vision": mean_vision,
        "mean_S_temporal": mean_temporal,
        "mean_S_physical": mean_physical,
        "mean_final_score": mean_final
    }

    final_results = [summary] + results

    # Save the evaluation results to a JSON file
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"Evaluation complete. Results saved to {save_path}")


# Process all datasets sequentially
for dataset_path, save_path in dataset_save_pairs:
    evaluate_dataset(dataset_path, save_path)