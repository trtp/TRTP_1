import os
import time
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# 清空 CUDA 缓存
torch.cuda.empty_cache()

# 加载模型
model_dir = "/media/ubuntu/10B4A468B4A451D0/models/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_dir)


def process_single_image(image_path, message_text):
    """
    处理单张图片并进行推理
    :param image_path: 图片文件路径
    :param message_text: 需要输入给模型的文本
    :return: 生成的文本
    """
    image = Image.open(image_path).convert("RGB")  # 确保是 RGB 格式
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": message_text}
        ]}
    ]

    # 处理输入
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors="pt").to(model.device)

    # 进行推理
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30)
        output_text = processor.decode(output[0], skip_special_tokens=True)

    return output_text


def batch_process_images(image_folder, output_folder, message_text):
    """
    批量处理文件夹中的所有图片
    :param image_folder: 源图片文件夹
    :param output_folder: 结果保存文件夹
    :param message_text: 需要输入给模型的文本
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, image_file)
        print(f"处理图片: {image_path} ({idx}/{len(image_files)})")

        # 推理单张图片
        output_text = process_single_image(image_path, message_text)

        # 保存结果
        result_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(output_text)

        print(f"结果已保存到: {result_file}")


# 示例用法
#image_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGet"  # 图片目录
image_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGetTest"  # 图片目录
output_folder = "/home/ubuntu/Desktop/dataset/droidCutImagePromptTest/Llama-3.2-11B-Vision-Instruct"  # 结果保存目录
message_text = "描述场景内物体的空间关系。"  # 提示词

start_time = time.time()
batch_process_images(image_folder, output_folder, message_text)
end_time = time.time()

print(f"批量处理耗时: {end_time - start_time:.2f} 秒")