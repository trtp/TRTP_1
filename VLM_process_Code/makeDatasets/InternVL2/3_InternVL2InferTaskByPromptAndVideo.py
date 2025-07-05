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
    Loads the large model and its tokenizer.
    :param model_path: The path to the model.
    :return: The model and tokenizer.
    """
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                      use_flash_attn=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    return model, tokenizer


def infer_task_planning(model, tokenizer, video_path, prompt, num_segments=8):
    """
    Infers the task plan from a video using the large model.
    :param model: The loaded large model.
    :param tokenizer: The model's tokenizer.
    :param video_path: The path to the video.
    :param prompt: The scene description prompt.
    :param num_segments: The number of segments.
    :return: The inference result as a string.
    """
    pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(len(num_patches_list))])
    question = prompt + video_prefix + 'List the action sequence of the robotic arm in the video'

    response, history = model.chat(tokenizer, pixel_values, question,
                                   generation_config=dict(max_new_tokens=1024, do_sample=True),
                                   num_patches_list=num_patches_list, history=None, return_history=True)

    return response


def update_dataset_with_inference(model, tokenizer, input_json, output_json):
    """
    Fills the 'gpt' field of a dataset with the inference results from the large model.
    :param model: The loaded large model.
    :param tokenizer: The model's tokenizer.
    :param input_json: The path to the input JSON file.
    :param output_json: The path to the output JSON file.
    """
    # Load the input dataset
    with open(input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Iterate through each item and perform inference
    for item in dataset:
        video_path = item["videos"][0]
        prompt = "This is a description of the spatial scene from the video frames, for your reference:" + \
                 item["conversations"][0]["value"]

        # Call the inference function
        print(f"Processing video: {video_path}")
        try:
            gpt_result = infer_task_planning(model, tokenizer, video_path, prompt)
            item["conversations"][2]["value"] = gpt_result
            print(f"Inference complete: {gpt_result}")
        except Exception as e:
            print(f"Inference failed for: {video_path}, Error: {e}")

    # Save the updated dataset
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"Updated dataset has been saved to: {output_json}")


if __name__ == "__main__":
    # Define paths
    model_path = '/media/ubuntu/10B4A468B4A451D0/models/InternVL2-8B'

    # Define a list of corresponding input_json and output_json pairs
    dataset_pairs = [
        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/llava-onevision-qwen2-7b-ov-hf-prompt-output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_llava-onevision-qwen2-7b-ov-hf-prompt.json"),
        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Ovis1.6-Gemma2-9B_prompt_output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_Ovis1.6-Gemma2-9B_prompt.json"),
        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Qwen2VL7B_prompt_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/InternVL2-8B_infer_Qwen2VL7B_prompt.json")
    ]

    # Load the model
    model, tokenizer = load_model(model_path)

    # Iterate through dataset_pairs and process each one
    for input_json, output_json in dataset_pairs:
        print(f"Processing: {input_json} -> {output_json}")
        update_dataset_with_inference(model, tokenizer, input_json, output_json)
        print(f"Finished processing: {input_json}\n")