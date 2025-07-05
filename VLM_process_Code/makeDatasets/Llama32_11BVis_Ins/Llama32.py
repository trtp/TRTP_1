# model_processor.py
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

def process_image(image_path, message_text):

    model_dir = "/media/ubuntu/10B4A468B4A451D0/models/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_dir)

    # 图像加载
    image = Image.open(image_path)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": message_text}
        ]}
    ]


    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors="pt").to(model.device)


    output = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(output[0])