import torch
from PIL import Image
from modelscope import AutoModelForCausalLM

# load model
model = AutoModelForCausalLM.from_pretrained("/media/ubuntu/10B4A468B4A451D0/models/Ovis1.6-Gemma2-9B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=8192,
                                             trust_remote_code=True).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()
#找窗口的ID，

# enter image path and prompt
image_path = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGet/video_2611_frame.jpg"
image = Image.open(image_path)
# text = "描述场景内物体的空间关系."# prompt
text = "请根据提供的图片，描述物体之间的空间关系，例如“物体A在物体B的左侧”或“物体C在物体D的上方”。"# prompt
query = f'<image>\n{text}'

# format conversation
prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
input_ids = input_ids.unsqueeze(0).to(device=model.device)
attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

# generate output
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
    print(f'Output:\n{output}')