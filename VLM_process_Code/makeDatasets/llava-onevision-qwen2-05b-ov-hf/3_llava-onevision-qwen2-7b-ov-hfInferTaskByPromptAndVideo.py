import os
import json
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

def load_model(model_path):
    """
    Loads the large model and its processor.
    :param model_path: The path to the model.
    :return: The model and processor.
    """
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def infer_task_planning(model, processor, system_value, human_value, video_path):
    """
    Infers the task plan from a video using the large model.
    :param model: The loaded large model.
    :param processor: The model's processor.
    :param system_value: The value from the 'system' field.
    :param human_value: The value from the 'human' field.
    :param video_path: The path to the video.
    :return: The inference result as a string.
    """
    # Construct the input message
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_value}]},
        {"role": "user", "content": [{"type": "text", "text": "<video>List the action sequence of the robotic arm. Be specific about the object being grasped, the action, and its orientation."}, {"type": "video", "video": video_path}]},
    ]

    # Prepare inputs for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda:0")  # Load input data to the main GPU

    # Inference
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return output_text


def update_dataset_with_inference(model, processor, input_json, output_json):
    """
    Fills the 'gpt' field of a dataset with the inference results from the large model.
    :param model: The loaded large model.
    :param processor: The model's processor.
    :param input_json: The path to the input JSON file.
    :param output_json: The path to the output JSON file.
    """
    # Load the input dataset
    with open(input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Iterate through each item and perform inference
    for item in dataset:
        system_value = item["conversations"][0]["value"]
        human_value = item["conversations"][1]["value"]
        video_path = item["videos"][0]

        # Call the inference function
        print(f"Processing video: {video_path}")
        try:
            gpt_result = infer_task_planning(model, processor, system_value, human_value, video_path)
            item["conversations"][2]["value"] = gpt_result
            print(f"Inference complete: {gpt_result}")
        except Exception as e:
            print(f"Inference failed for: {video_path}, Error: {e}")

    # Save the updated dataset
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"Updated dataset has been saved to: {output_json}")


if __name__ == "__main__":
    # Define the new model path (if different)
    model_path = "/media/ubuntu/10B4A468B4A451D0/models/llava-onevision-qwen2-7b-ov-hf"

    # Datasets for batch inference
    dataset_pairs = [
        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/InternVL2-8B-prompt-output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_InternVL2-8B-prompt.json"),
        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/InternVL2_5-8B-prompt-output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_InternVL2_5-8B-prompt.json"),
        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Llama-3.2-11B-prompt-output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_Llama-3.2-11B-_prompt.json"),
        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/llava-onevision-qwen2-7b-ov-hf-prompt-output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_llava-onevision-qwen2-7b-ov-hf-prompt.json"),
        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/llava-v1.6-vicuna-7b-hf-prompt-output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_llava-v1.6-vicuna-7b-hf-prompt.json"),
        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/MiniCPM-V-2_prompt_6output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_MiniCPM-V-2_prompt.json"),
        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Molmo-7B-D-0924-prompt-output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_Molmo-7B-D-0924-prompt.json"),
        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Ovis1.6-Gemma2-9B_prompt_output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_Ovis1.6-Gemma2-9B_prompt.json"),
        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Qwen2VL7B_prompt_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_Qwen2VL7B_prompt.json")
    ]

    # Load the model
    model, processor = load_model(model_path)

    # Iterate through dataset_pairs and process each one
    for input_json, output_json in dataset_pairs:
        print(f"Processing: {input_json} -> {output_json}")
        update_dataset_with_inference(model, processor, input_json, output_json)
        print(f"Finished processing: {input_json}\n")