import os
import json
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

def load_model(model_path):
    """
    Loads the large model and its processor.
    :param model_path: The path to the model.
    :return: The model and processor.
    """
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        offload_folder="offload",
        offload_state_dict=True,
    )
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
        # {"role": "user", "content": [{"type": "text", "text": human_value}, {"type": "video", "video": video_path}]},
        {"role": "user", "content": [{"type": "text", "text": "<video>List the action sequence of the robotic arm in the video"}, {"type": "video", "fps": 1.0, "video": video_path}]},
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
        generated_ids = model.generate(**inputs, max_new_tokens=256)
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


def get_model_name(model_path):
    """Extracts the model name from the model path"""
    return os.path.basename(model_path)


if __name__ == "__main__":
    # List of model paths
    model_paths = [
        "/home/ubuntu/Desktop/LLaMA-Factory/src/saves/Qwen2-VL-7B-Instruct/lora/merge/MiniCPM-V-2_6_infer",
    ]

    # Input and output JSON datasets
    dataset_pairs = [
        ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/Ovis1.6-Gemma2-9B_prompt_output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output")  # Output folder path
    ]

    # Iterate through datasets and models
    for input_json, output_folder in dataset_pairs:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print(f"Processing dataset: {input_json}")

        for model_path in tqdm(model_paths, desc="Model Inference"):
            model_name = get_model_name(model_path)

            # Output filename format: model_name_qwen2vlsft.json
            output_json = os.path.join(output_folder, f"{model_name}_qwen2vlsft.json")

            print(f"\nUsing model: {model_name}")

            # Load the model
            model, tokenizer = load_model(model_path)

            # Perform inference and save the output
            update_dataset_with_inference(model, tokenizer, input_json, output_json)

            print(f"Finished processing with {model_name}: {output_json}")

    print("\n All models have finished processing.")