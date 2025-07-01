import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import time

# 清空 CUDA 缓存
torch.cuda.empty_cache()

# 加载模型路径
model_path = "/home/ubuntu/Desktop/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",  # 自动分配到多张显卡
    offload_folder="offload",  # 如果显存仍然不足，部分参数会移到 CPU
    offload_state_dict=True,  # 开启参数的分阶段加载
)

# 加载 Processor
processor = AutoProcessor.from_pretrained(model_path)

def process_single_image(image_path):
    """
    处理单张图片并推理位置信息。
    :param image_path: 图片文件路径。
    :return: 推理结果文本。
    """
    messages = [

        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述图片中物体之间的位置关系"},
                #{"type": "text", "text": "请分析图片中物体的相对位置，如‘物体 A 在物体 B 的左侧’等。"},
                {"type": "image", "image": image_path},
            ],
        }
    ]
    # 准备推理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda:0")  # 将输入数据加载到主 GPU

    # 推理
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
#image_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGet"  # 替换为包含图片的文件夹路径
image_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGetTest"  # 替换为包含图片的文件夹路径
output_folder = "/home/ubuntu/Desktop/dataset/droidCutImagePrompt/Qwen2VL7B"  # 替换为保存结果的文件夹路径

start_time = time.time()  # 开始计时
batch_process_images(image_folder, output_folder)
end_time = time.time()  # 结束计时
print(f"批量处理耗时: {end_time - start_time:.2f} 秒")