import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from modelscope import snapshot_download
# model_id = "LLM-Research/Llama-3.2-11B-Vision-Instruct"
# model_dir = snapshot_download(model_id, ignore_file_pattern=['*.pth'])
model_dir = "/media/ubuntu/10B4A468B4A451D0/models/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_dir)

# url = "https://www.modelscope.cn/models/LLM-Research/Llama-3.2-11B-Vision/resolve/master/rabbit.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# image="/home/ubuntu/Desktop/LLaMA-Factory/assets/4.jpg"
image_path ="/home/ubuntu/Desktop/dataset/droidCutImage_randomGet/video_2611_frame.jpg"
image = Image.open(image_path)  # 确保返回的是 PIL.Image.Image 对象
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "描述图片中物体之间的空间位置关系"}
        #{"type": "text", "text": "请分析图片中物体的相对位置，如‘物体 A 在物体 B 的左侧’等。"}
        #{"type": "text", "text": "请按照‘物体 A 在物体 B 的[左侧/右侧/前方/后方/上方/下方]’的格式描述图片中的物体关系。"}
    ]}
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(output[0]))