import json
import cv2
import torch
import time
import re
import os
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 1. Load the model
model_path = "/home/ubuntu/Desktop/Qwen2-VL-2B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    ignore_mismatched_sizes=True,
).to(device)
processor = AutoProcessor.from_pretrained(model_path)

# 2. Load the dataset
dataset_path = "/home/ubuntu/Desktop/dataset/droidJsonDatset/InterVL2output_dataset.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

results_file = "evaluation_results.json"

# If a results file already exists, load the previous results and cumulative score data
if os.path.exists(results_file):
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        individual_results = existing_data.get("individual_results", [])
        # Recalculate cumulative scores and the count of valid scores
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
    except (json.JSONDecodeError, FileNotFoundError):
        individual_results = []
        precision_sum = 0.0
        completeness_sum = 0.0
        redundancy_sum = 0.0
        valid_count = 0
else:
    individual_results = []
    precision_sum = 0.0
    completeness_sum = 0.0
    redundancy_sum = 0.0
    valid_count = 0

# 3. Iterate through the dataset for evaluation
for data in dataset:
    system_value = data["conversations"][0]["value"]  # Ground Truth
    video_path = data["videos"][0]  # Video path

    # Check if this video has already been processed
    if any(res['video'] == video_path for res in individual_results):
        print(f"Skipping already processed video: {video_path}")
        continue

    # Read the 10th frame of the video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Could not read the 10th frame from {video_path}")
        continue

    # Save a temporary image
    temp_image_path = "temp_frame.jpg"
    cv2.imwrite(temp_image_path, frame)

    # Construct the conversation message payload
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please strictly evaluate the ability to extract spatial relationships from the given image and text description, and quantify the following three metrics:"},
                {"type": "text", "text": "1. **Precision**: The proportion of correct relationships among the predicted ones, avoiding excessive incorrect information."},
                {"type": "text", "text": "2. **Completeness**: Whether the descriptions include the spatial relationships of all key objects."},
                {"type": "text", "text": "3. **Redundancy**: Whether there are too many irrelevant descriptions that hinder comprehension efficiency."},
                {"type": "text", "text": "Please provide a quantitative score (0-100 for each) based on the image content and the following Ground Truth:"},
                {"type": "text", "text": f"Ground Truth: {system_value}"},
                {"type": "image", "image": temp_image_path},
            ],
        }
    ]

    # 4. Perform inference
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

    # Parse the output text using regular expressions to extract scores
    pattern = r"Precision:\s*([\d\.]+).*Completeness:\s*([\d\.]+).*Redundancy:\s*([\d\.]+)"
    match = re.search(pattern, output_text, re.S | re.IGNORECASE)
    if match:
        precision = round(float(match.group(1)), 3)
        completeness = round(float(match.group(2)), 3)
        redundancy = round(float(match.group(3)), 3)
        # Accumulate valid scores
        precision_sum += precision
        completeness_sum += completeness
        redundancy_sum += redundancy
        valid_count += 1
    else:
        print(f"Could not parse the output for {video_path}")
        precision = None
        completeness = None
        redundancy = None

    # Build the result for the current sample
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

    # Recalculate the averages
    if valid_count > 0:
        avg_precision = round(precision_sum / valid_count, 3)
        avg_completeness = round(completeness_sum / valid_count, 3)
        avg_redundancy = round(redundancy_sum / valid_count, 3)
    else:
        avg_precision = avg_completeness = avg_redundancy = None

    # Construct the data dictionary to be saved
    output_data = {
        "individual_results": individual_results,
        "average_scores": {
            "Precision": avg_precision,
            "Completeness": avg_completeness,
            "Redundancy": avg_redundancy
        }
    }

    # Write to the save file after processing each sample (overwrite)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Finished processing {video_path}, output: {output_text}")

# 5. Print the final average scores
print("\n===== Evaluation Averages =====")
print(f"Precision: {avg_precision:.2f}" if avg_precision is not None else "Precision: N/A")
print(f"Completeness: {avg_completeness:.2f}" if avg_completeness is not None else "Completeness: N/A")
print(f"Redundancy: {avg_redundancy:.2f}" if avg_redundancy is not None else "Redundancy: N/A")
print("Evaluation complete. Results have been saved to evaluation_results.json")