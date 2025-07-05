import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import time

# Clear CUDA cache
torch.cuda.empty_cache()

# Load model path
model_path = "/home/ubuntu/Desktop/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",          # Automatically distribute to multiple GPUs
    offload_folder="offload",   # If VRAM is still insufficient, some parameters will be moved to the CPU
    offload_state_dict=True,    # Enable staged loading of parameters
)

# Load Processor
processor = AutoProcessor.from_pretrained(model_path)

def process_single_image(image_path):
    """
    Processes a single image and infers positional information.
    :param image_path: Path to the image file.
    :return: The inference result text.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the positional relationships between objects in the image"},
                #{"type": "text", "text": "Please analyze the relative positions of objects in the image, such as 'object A is to the left of object B', etc."},
                {"type": "image", "image": image_path},
            ],
        }
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
        )
    return output_text[0]

def batch_process_images(image_folder, output_folder):
    """
    Batch processes all images in a folder and saves the inference results.
    :param image_folder: The source folder containing images.
    :param output_folder: The target folder to save inference results.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing image: {image_path} ({idx}/{len(image_files)})")

        # Infer a single image
        output_text = process_single_image(image_path)

        # Save the result to a text file
        result_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(output_text)

        print(f"Result saved to: {result_file}")

# Example usage
# image_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGet"  # Replace with the folder containing images
image_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGetTest"  # Replace with the folder containing images
output_folder = "/home/ubuntu/Desktop/dataset/droidCutImagePrompt/Qwen2VL7B"  # Replace with the folder to save results

start_time = time.time()  # Start timing
batch_process_images(image_folder, output_folder)
end_time = time.time()  # End timing
print(f"Batch processing time: {end_time - start_time:.2f} seconds")