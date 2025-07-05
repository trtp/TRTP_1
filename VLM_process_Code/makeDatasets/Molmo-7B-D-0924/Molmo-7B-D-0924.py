from modelscope import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests

# load the processor
processor = AutoProcessor.from_pretrained(
    '/media/ubuntu/10B4A468B4A451D0/models/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    '/media/ubuntu/10B4A468B4A451D0/models/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)
image1 = Image.open('/home/ubuntu/Desktop/dataset/droidCutImage_randomGet/video_2611_frame.jpg').convert('RGB')
# process the image and text
# inputs = processor.process(
#     images=[image1],
#     text="Describe the objects and their spatial relationships within the scene."
# )
inputs = processor.process(
    images=[image1],
    text="Please describe the spatial relationship between objects based on the provided images, such as 'Object A is to the left of Object B' or 'Object C is above Object D."
)


# move inputs to the correct device and make a batch of size 1
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

# generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer
)

# only get generated tokens; decode them to text
generated_tokens = output[0,inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

# print the generated text
print(generated_text)

