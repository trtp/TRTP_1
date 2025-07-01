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


# 初始化模型
model_path = '/media/ubuntu/10B4A468B4A451D0/models/InternVL2-8B'
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
    处理单张图片并推理位置信息。
    :param image_path: 图片文件路径。
    :return: 推理结果文本。
    """
    pixel_values = load_image(image_path).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    question = '<image>\n描述场景内物体的空间关系.'
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response


def batch_process_images(image_folder, output_folder):
    """
    批量处理文件夹中的所有图片，并保存推理结果。
    :param image_folder: 包含图片的源文件夹。
    :param output_folder: 结果保存路径。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, image_file)
        print(f"处理图片: {image_path} ({idx}/{len(image_files)})")

        # 推理
        output_text = process_single_image(image_path)

        # 保存结果
        result_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(output_text)

        print(f"结果已保存到: {result_file}")


# 使用示例
image_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGet"
output_folder = "/home/ubuntu/Desktop/dataset/droidCutImagePrompt/InternVL2-8B"

batch_process_images(image_folder, output_folder)