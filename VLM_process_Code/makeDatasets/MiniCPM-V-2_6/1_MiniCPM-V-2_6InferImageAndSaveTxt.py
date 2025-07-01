import os
import time
import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer

# 加载模型
model_path = "/media/ubuntu/10B4A468B4A451D0/models/MiniCPM-V-2_6"
model = AutoModel.from_pretrained(
    model_path, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def process_single_image(image_path):
    """
    处理单张图片并推理位置信息。
    :param image_path: 图片文件路径。
    :return: 推理结果文本。
    """
    image = Image.open(image_path).convert('RGB')
    question = '描述场景内物体的空间关系.'
    msgs = [{'role': 'user', 'content': [image, question]}]

    # 推理
    answer = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
    return answer


def batch_process_images(image_folder, output_folder):
    """
    批量处理一个文件夹中的所有图片，并保存推理结果。
    :param image_folder: 包含图片的源文件夹。
    :param output_folder: 推理结果保存的目标文件夹。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, image_file)
        print(f"处理图片: {image_path} ({idx}/{len(image_files)})")

        # 推理单张图片
        output_text = process_single_image(image_path)

        # 保存结果到文本文件
        result_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(output_text)

        print(f"结果已保存到: {result_file}")


# 使用示例
image_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGet"  # 替换为包含图片的文件夹路径
output_folder = "/home/ubuntu/Desktop/dataset/droidCutImagePrompt/MiniCPM-V-2_6"  # 替换为保存结果的文件夹路径

start_time = time.time()
batch_process_images(image_folder, output_folder)
end_time = time.time()

print(f"批量处理耗时: {end_time - start_time:.2f} 秒")