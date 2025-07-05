import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('/media/ubuntu/10B4A468B4A451D0/models/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('/media/ubuntu/10B4A468B4A451D0/models/MiniCPM-V-2_6', trust_remote_code=True)

image1 = Image.open('/home/ubuntu/Desktop/dataset/droidCutImage_randomGet/video_2611_frame.jpg').convert('RGB')

question = 'Describe the objects and their spatial relationships within the scene.'

msgs = [{'role': 'user', 'content': [image1, question]}]

answer = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)