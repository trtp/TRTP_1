from vllm import LLM
from vllm.sampling_params import SamplingParams
from modelscope import snapshot_download

model_name = "/media/ubuntu/10B4A468B4A451D0/models/Pixtral-12B-2409"
max_img_per_msg = 5
max_tokens_per_img = 4096

sampling_params = SamplingParams(max_tokens=8192, temperature=0.7)
llm = LLM(model=model_name, tokenizer_mode="mistral", limit_mm_per_prompt={"image": max_img_per_msg}, max_num_batched_tokens=max_img_per_msg * max_tokens_per_img)

prompt = "Describe the following image."

url_1 = "https://model-demo.oss-cn-hangzhou.aliyuncs.com/demo.jpeg"
url_2 = "https://picsum.photos/seed/picsum/200/300"
url_3 = "https://picsum.photos/id/32/512/512"

messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url_1}}, {"type": "image_url", "image_url": {"url": url_2}}],
    }
]

outputs = llm.chat(messages=messages, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)