import os
import time
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# Clear CUDA cache
torch.cuda.empty_cache()

# Load the model
model_dir = "/media/ubuntu/10B4A468B4A451D0/models/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_dir)


def process_single_image(image_path, message_text):
    """
    Processes a single image and performs inference.
    :param image_path: Path to the image file.
    :param message_text: The text to be input to the model.
    :return: The generated text.
    """
    image = Image.open(image_path).convert("RGB")  # Ensure it is in RGB format
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": message_text}
        ]}
    ]

    # Process the inputs
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors="pt").to(model.device)

    # Perform inference
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30)
        output_text = processor.decode(output[0], skip_special_tokens=True)

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
# image_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGet"  # Image directory
image_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGetTest"  # Image directory
output_folder = "/home/ubuntu/Desktop/dataset/droidCutImagePromptTest/Llama-3.2-11B-Vision-Instruct"  # Directory to save results
message_text = "Describe the spatial relationships of the objects in the scene."  # Prompt text

start_time = time.time()
batch_process_images(image_folder, output_folder, message_text)
end_time = time.time()

print(f"Batch processing time: {end_time - start_time:.2f} seconds")