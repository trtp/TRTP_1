import torch
from PIL import Image
from modelscope import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("/media/ubuntu/10B4A468B4A451D0/models/Ovis1.6-Gemma2-9B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=8192,
                                             trust_remote_code=True).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()
# To find the window ID, use `xwininfo -tree -root` in the terminal.

# Enter image path and prompt
image_path = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGet/video_2611_frame.jpg"
image = Image.open(image_path)
# text = "Describe the spatial relationships of the objects in the scene."  # prompt
text = "Based on the provided image, describe the spatial relationships between objects, for example, 'object A is to the left of object B' or 'object C is above object D'." # prompt
query = f'<image>\n{text}'

# Format conversation
prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
input_ids = input_ids.unsqueeze(0).to(device=model.device)
attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

# Generate output
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