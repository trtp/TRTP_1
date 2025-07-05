import os
import time
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# Clear CUDA cache
torch.cuda.empty_cache()

# Load the model
model_id = "/media/ubuntu/10B4A468B4A451D0/models/llava-onevision-qwen2-7b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

def process_single_image(image_path, message_text):
    """
    Processes a single image and performs inference.
    :param image_path: Path to the image file.
    :param message_text: The text to be input to the model.
    :return: The generated text.
    """
    raw_image = Image.open(image_path).convert("RGB")  # Ensure it is in RGB format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": message_text},
                {"type": "image"},
            ],
        },
    ]

    # Process the inputs
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

    # Perform inference
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        output_text = processor.decode(output[0][2:], skip_special_tokens=True)

    return output_text

def batch_process_images(image_folder, output_folder, message_text):
    """
    Batch processes all images in a folder.
    :param image_folder: The source image folder.
    :param output_folder: The folder to save the results.
    :param message_text: The text to be input to the model.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing image: {image_path} ({idx}/{len(image_files)})")

        # Infer a single image
        output_text = process_single_image(image_path, message_text)

        # Save the result
        result_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(output_text)

        print(f"Result saved to: {result_file}")

# Example usage
image_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGet"  # Image directory
output_folder = "/home/ubuntu/Desktop/dataset/droidCutImagePrompt/llava-onevision-qwen2-7b-ov-hf"  # Directory to save results
message_text = "Describe the spatial relationships of the objects in the scene."  # Prompt for the model

start_time = time.time()
batch_process_images(image_folder, output_folder, message_text)
end_time = time.time()

print(f"Batch processing time: {end_time - start_time:.2f} seconds")