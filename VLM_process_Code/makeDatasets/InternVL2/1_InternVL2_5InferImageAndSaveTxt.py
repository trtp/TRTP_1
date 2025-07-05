import os
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from modelscope import AutoModel, AutoTokenizer


def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    pixel_values = transform(image).unsqueeze(0)
    return pixel_values


# Initialize the model
model_path = '/media/ubuntu/10B4A468B4A451D0/models/InternVL2_5-8B'
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)


def process_single_image(image_path):
    """
    Process a single image and infer spatial information.
    :param image_path: The path to the image file.
    :return: The inference result as text.
    """
    pixel_values = load_image(image_path).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    # The original prompt was: '<image>\n描述场景内物体的空间关系.'
    question = '<image>\nDescribe the spatial relationships of the objects in the scene.'
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response


def batch_process_images(image_folder, output_folder):
    """
    Batch process all images in a folder and save the inference results.
    :param image_folder: The source folder containing the images.
    :param output_folder: The path to save the results.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing image: {image_path} ({idx}/{len(image_files)})")

        # Inference
        output_text = process_single_image(image_path)

        # Save the result
        result_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(output_text)

        print(f"Result saved to: {result_file}")


# Usage Example
image_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGet"
output_folder = "/home/ubuntu/Desktop/dataset/droidCutImagePrompt/InternVL2_5-8B"

batch_process_images(image_folder, output_folder)