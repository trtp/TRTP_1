import os
import torch
from PIL import Image
from modelscope import AutoModelForCausalLM
import time

# 加载模型
model = AutoModelForCausalLM.from_pretrained("/media/ubuntu/10B4A468B4A451D0/models/Ovis1.6-Gemma2-9B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=8192,
                                             trust_remote_code=True).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

def process_single_image(image_path, text_prompt):
    """
    处理单张图片并生成描述文本。
    :param image_path: 图片文件路径。
    :param text_prompt: 文字提示，用于生成图像描述。
    :return: 生成的文本描述。
    """
    image = Image.open(image_path)
    query = f'<image>\n{text_prompt}'

    # 格式化输入
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

    # 推理生成输出
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
    return output

def batch_process_images(image_folder, output_folder, text_prompt):
    """
    批量处理文件夹中的所有图片并保存推理结果。
    :param image_folder: 包含图片的源文件夹。
    :param output_folder: 推理结果保存的目标文件夹。
    :param text_prompt: 提示文本，用于生成图片描述。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, image_file)
        print(f"处理图片: {image_path} ({idx}/{len(image_files)})")

        # 推理单张图片
        output_text = process_single_image(image_path, text_prompt)

        # 保存结果到文本文件
        result_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(output_text)

        print(f"结果已保存到: {result_file}")

# 使用示例
image_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGetTest"  # 替换为包含图片的文件夹路径
output_folder = "/home/ubuntu/Desktop/dataset/droidCutImagePromptTest/Ovis1.6-Gemma2-9B"  # 替换为保存结果的文件夹路径
#text_prompt = "请根据提供的图片，描述物体之间的空间关系，例如“物体A在物体B的左侧”或“物体C在物体D的上方”。"
text_prompt = "请根据提供的图片，描述物体之间的空间关系。"

start_time = time.time()  # 开始计时
batch_process_images(image_folder, output_folder, text_prompt)
end_time = time.time()  # 结束计时
print(f"批量处理耗时: {end_time - start_time:.2f} 秒")