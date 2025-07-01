import os
import torch
from modelscope import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import time

# 清空 CUDA 缓存
torch.cuda.empty_cache()

# 加载模型路径
model_path = "/media/ubuntu/10B4A468B4A451D0/models/Molmo-7B-D-0924"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# 加载 Processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


def process_single_image(image_path):
    """
    处理单张图片并推理位置信息。
    :param image_path: 图片文件路径。
    :return: 推理结果文本。
    """
    image = Image.open(image_path).convert('RGB')
    inputs = processor.process(
        images=[image],
        #text="请根据提供的图片，描述物体之间的空间关系，例如‘物体A在物体B的左侧’或‘物体C在物体D的上方’。"
    text = "请根据提供的图片，描述物体之间的空间关系。"
    )

    # 移动输入到模型设备
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # 生成推理结果
    with torch.no_grad():
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

    # 解析输出
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    output_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return output_text


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
output_folder = "/home/ubuntu/Desktop/dataset/droidCutImagePrompt/Molmo-7B-D-0924"  # 替换为保存结果的文件夹路径

start_time = time.time()  # 开始计时
batch_process_images(image_folder, output_folder)
end_time = time.time()  # 结束计时
print(f"批量处理耗时: {end_time - start_time:.2f} 秒")