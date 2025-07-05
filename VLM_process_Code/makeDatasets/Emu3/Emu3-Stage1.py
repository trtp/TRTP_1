from PIL import Image
from modelscope import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, \
    UnbatchedClassifierFreeGuidanceLogitsProcessor
import torch

from modelscope import snapshot_download

# model path
# EMU_HUB = snapshot_download("BAAI/Emu3-Stage1")
VQ_HUB = snapshot_download("BAAI/Emu3-VisionTokenizer")
model_path = "/media/ubuntu/10B4A468B4A451D0/models/Emu3-Stage1"
import sys

# sys.path.append(EMU_HUB)
from VLM_process_Code.makeDatasets.Emu3.emu3.mllm.processing_emu3 import Emu3Processor

# prepare model and processor
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="balanced",
    offload_folder="offload",
    # attn_implementation="flash_attention_2",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda:0", trust_remote_code=True).eval()
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer, chat_template="{image_prompt}{text_prompt}")

# Image Generation
# prepare input
POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

classifier_free_guidance = 3.0
prompt = "a portrait of young girl."
prompt += POSITIVE_PROMPT

kwargs = dict(
    mode='G',
    ratio="1:1",
    image_area=model.config.image_area,
    return_tensors="pt",
    padding="longest",
)

pos_inputs = processor(text=prompt, **kwargs)
neg_inputs = processor(text=NEGATIVE_PROMPT, **kwargs)

# prepare hyper parameters
GENERATION_CONFIG = GenerationConfig(
    use_cache=True,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.pad_token_id,
    max_new_tokens=40960,
    do_sample=True,
    top_k=2048,
)

h = pos_inputs.image_size[:, 0]
w = pos_inputs.image_size[:, 1]
constrained_fn = processor.build_prefix_constrained_fn(h, w)
logits_processor = LogitsProcessorList([
    UnbatchedClassifierFreeGuidanceLogitsProcessor(
        classifier_free_guidance,
        model,
        unconditional_ids=neg_inputs.input_ids.to("cuda:0"),
    ),
    PrefixConstrainedLogitsProcessor(
        constrained_fn,
        num_beams=1,
    ),
])

# generate
outputs = model.generate(
    pos_inputs.input_ids.to("cuda:0"),
    GENERATION_CONFIG,
    logits_processor=logits_processor,
    attention_mask=pos_inputs.attention_mask.to("cuda:0"),
)

mm_list = processor.decode(outputs[0])
for idx, im in enumerate(mm_list):
    if not isinstance(im, Image.Image):
        continue
    im.save(f"result_{idx}.png")

# Multimodal Understanding
text = "The image depicts "
image = Image.open("/home/ubuntu/Desktop/LLaMA-Factory/assets/4.jpg")
inputs = processor(
    text=text,
    image=image,
    mode='U',
    padding="longest",
    return_tensors="pt",
)

GENERATION_CONFIG = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=1024,
)

outputs = model.generate(
    inputs.input_ids.to("cuda:0"),
    GENERATION_CONFIG,
    attention_mask=inputs.attention_mask.to("cuda:0"),
)

outputs = outputs[:, inputs.input_ids.shape[-1]:]
answers = processor.batch_decode(outputs, skip_special_tokens=True)
for ans in answers:
    print(ans)