from PIL import Image
import requests
from modelscope import AutoModelForCausalLM
from modelscope import AutoProcessor
from modelscope import snapshot_download

model_id = "/media/ubuntu/10B4A468B4A451D0/models/Phi-3.5-vision-instruct"

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  device_map="cuda",
  trust_remote_code=True,
  torch_dtype="auto",
  # _attn_implementation='flash_attention_2'
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id,
  trust_remote_code=True,
  num_crops=4
)

images = []
placeholder = ""

# Note: if OOM, you might consider reduce number of frames in this example.
# for i in range(1,20):
#     url = f"https://modelscope.oss-cn-beijing.aliyuncs.com/resource/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.webp"
#     images.append(Image.open(requests.get(url, stream=True).raw))
#     placeholder += f"<|image_{i}|>\n"
image_path = "/home/ubuntu/Desktop/LLaMA-Factory/assets/4.jpg"
image = Image.open(image_path)
images.append(image)
messages = [
    {"role": "user", "content": placeholder+"Summarize the deck of slides."},
]

prompt = processor.tokenizer.apply_chat_template(
  messages,
  tokenize=False,
  add_generation_prompt=True
)

inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

generation_args = {
    "max_new_tokens": 1000,
    "temperature": 0.0,
    "do_sample": False,
}

generate_ids = model.generate(**inputs,
  eos_token_id=processor.tokenizer.eos_token_id,
  **generation_args
)

# remove input tokens
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids,
  skip_special_tokens=True,
  clean_up_tokenization_spaces=False)[0]

print(response)