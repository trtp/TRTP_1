import os
import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import time

# Clear CUDA cache
torch.cuda.empty_cache()

# Load model path
model_path = "/media/ubuntu/10B4A468B4A451D0/models/llava-v1.6-vicuna-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_path)

model = LlavaNextForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.to("cuda:0")


def process_single_image(image_path):
    """
    Processes a single image and infers a description.
    :param image_path: Path to the image file.
    :return: The inference result text.
    """
    image = Image.open(image_path)

    prompt = "<image>Describe the spatial relationships of the objects in the scene."

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    # Inference
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
        output_text = processor.decode(output[0], skip_special_tokens=True)

    return output_text


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
image_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGet"  # Replace with your folder containing images
output_folder = "/home/ubuntu/Desktop/dataset/droidCutImagePrompt/llava-v1.6-vicuna-7b-hf"  # Replace with the folder to save results

start_time = time.time()  # Start timing
batch_process_images(image_folder, output_folder)
end_time = time.time()  # End timing
print(f"Batch processing time: {end_time - start_time:.2f} seconds")