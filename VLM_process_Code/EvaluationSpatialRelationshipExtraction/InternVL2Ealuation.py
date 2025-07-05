import json
import os
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from modelscope import AutoModel, AutoTokenizer

# Dataset paths (multiple lines are preserved as in the original)
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
    """Loads a specific frame from a video (e.g., the 10th frame)."""
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    frame_idx = min(frame_idx, total_frames - 1)  # Prevent index from going out of bounds
    frame = vr[frame_idx].asnumpy()
    return Image.fromarray(frame)


def load_image(image, input_size=448):
    """Preprocesses an image and converts it to the model's input format."""
    transform = build_transform(input_size=input_size)
    return transform(image).unsqueeze(0)  # Add batch dimension


def evaluate_spatial_relationship(video_path, system_description, model, tokenizer):
    """Evaluates the spatial relationship match between an image and a system description."""
    # Load the 10th frame
    image = load_video_frame(video_path, frame_idx=10)

    # Preprocess the image
    pixel_values = load_image(image).to(torch.bfloat16).cuda()

    # Generate the input prompt
    question = (f"<image>\n{system_description}\nBased on the image and the input description, evaluate the spatial "
                f"relationship in terms of precision, completeness, and redundancy, and provide a score (0-1 for each). "
                f"Please strictly return the scores in the following format: "
                f"Precision: <value> Completeness: <value> Redundancy: <value>")

    # Inference
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    response = model.chat(tokenizer, pixel_values, question, generation_config)

    # Parse the model's output
    print(f"Model response: {response}")

    # Parse the Precision, Completeness, and Redundancy scores from the result
    scores = {"Precision": None, "Completeness": None, "Redundancy": None}
    for key in scores.keys():
        try:
            score_str = response.split(key + ":")[1].split()[0]
            score = float(score_str)
            scores[key] = round(score, 3)  # Round to 3 decimal places
        except (IndexError, ValueError) as e:
            scores[key] = None  # Set to None if parsing fails

    return scores


# Load the Vision-Language Model
path = "/media/ubuntu/10B4A468B4A451D0/models/InternVL2-8B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# Load the dataset
with open(DATASET_PATH, "r") as f:
    dataset = json.load(f)

# Iterate through the dataset for evaluation
for idx, sample in enumerate(dataset):
    system_text = sample["conversations"][0]["value"]
    video_path = sample["videos"][0]

    print(f"Evaluating {idx + 1}/{len(dataset)}, video: {video_path}")
    scores = evaluate_spatial_relationship(video_path, system_text, model, tokenizer)

    # Build the result for the current sample
    current_result = {
        "video": video_path,
        "Precision": scores["Precision"],
        "Completeness": scores["Completeness"],
        "Redundancy": scores["Redundancy"]
    }

    # If the results file exists, load existing results; otherwise, initialize.
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r") as f:
                output_data = json.load(f)
            individual_results = output_data.get("individual_results", [])
        except json.JSONDecodeError:
            individual_results = [] # Handle empty or corrupt file
    else:
        individual_results = []

    # Add the current result to the results list
    individual_results.append(current_result)

    # Recalculate the averages based on all individual_results
    total_scores = {"Precision": 0.0, "Completeness": 0.0, "Redundancy": 0.0}
    valid_count = {"Precision": 0, "Completeness": 0, "Redundancy": 0}
    for res in individual_results:
        for key in total_scores.keys():
            if isinstance(res.get(key), (int, float)):
                total_scores[key] += res[key]
                valid_count[key] += 1

    average_scores = {}
    for key in total_scores.keys():
        if valid_count[key] > 0:
            average_scores[key] = round(total_scores[key] / valid_count[key], 3)
        else:
            average_scores[key] = "N/A"

    # Construct the dictionary to be saved
    output_data = {
        "individual_results": individual_results,
        "average_scores": average_scores
    }

    # Write back to the file, overwriting it (this saves all results each time)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Current evaluation results have been saved to: {RESULTS_PATH}\n")