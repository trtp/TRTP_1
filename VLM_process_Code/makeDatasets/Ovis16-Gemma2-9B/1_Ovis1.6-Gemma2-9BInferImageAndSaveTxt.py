import os
import torch
from PIL import Image
from modelscope import AutoModelForCausalLM
import time

# Load the model
model = AutoModelForCausalLM.from_pretrained("/media/ubuntu/10B4A468B4A451D0/models/Ovis1.6-Gemma2-9B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=8192,
                                             trust_remote_code=True).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

def process_single_image(image_path, text_prompt):
    """
    Processes a single image and generates a descriptive text.
    :param image_path: Path to the image file.
    :param text_prompt: Text prompt used to generate the image description.
    :return: The generated text description.
    """
    image = Image.open(image_path)
    query = f'<image>\n{text_prompt}'

    # Format the inputs
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, [image])
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

    # Infer and generate output
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
    Batch processes all images in a folder and saves the inference results.
    :param image_folder: The source folder containing images.
    :param output_folder: The target folder to save inference results.
    :param text_prompt: The prompt text used to generate image descriptions.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing image: {image_path} ({idx}/{len(image_files)})")

        # Infer a single image
        output_text = process_single_image(image_path, text_prompt)

        # Save the result to a text file
        result_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(output_text)

        print(f"Result saved to: {result_file}")

# Example usage
image_folder = "/home/ubuntu/Desktop/dataset/droidCutImage_randomGetTest"  # Replace with the folder containing images
output_folder = "/home/ubuntu/Desktop/dataset/droidCutImagePromptTest/Ovis1.6-Gemma2-9B"  # Replace with the folder to save results
text_prompt = "Please describe the spatial relationships between objects based on the provided image."

start_time = time.time()  # Start timing
batch_process_images(image_folder, output_folder, text_prompt)
end_time = time.time()  # End timing
print(f"Batch processing time: {end_time - start_time:.2f} seconds")