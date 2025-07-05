import numpy as np
import torch
import torchvision.transforms as T
import pygame
import pygame.camera
import time
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from modelscope import AutoModel, AutoTokenizer

# Define constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Camera resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# INTERNVL_MODEL_PATH = '/media/ubuntu/10B4A468B4A451D0/models/InternVL2_5-8B'
INTERNVL_MODEL_PATH = '/media/ubuntu/10B4A468B4A451D0/models/InternVL2-8B'

# Preprocessing
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

# Dynamic preprocessing
def dynamic_preprocess(image, image_size=448, max_num=12):
    orig_width, orig_height = image.size
    resized_img = image.resize((image_size, image_size))
    return [resized_img]  # Simplified to a single image here, no slicing

# Process camera image
def preprocess_camera_image(image_surface, input_size=448):
    image_rgb = pygame.surfarray.array3d(image_surface)  # Convert to a numpy array
    image_pil = Image.fromarray(image_rgb.transpose(1, 0, 2))  # Convert to a PIL.Image
    processed_images = dynamic_preprocess(image_pil, image_size=input_size)
    transform = build_transform(input_size=input_size)
    pixel_values = [transform(image) for image in processed_images]
    return torch.stack(pixel_values).to(torch.bfloat16).cuda()

# Load InternVL2 model
print("Loading model...")
model = AutoModel.from_pretrained(
    INTERNVL_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(INTERNVL_MODEL_PATH, trust_remote_code=True, use_fast=False)
print("Model loaded successfully!")

# Initialize the camera
pygame.init()
pygame.camera.init()
cam_list = pygame.camera.list_cameras()
if not cam_list:
    print("No camera found")
    exit()
cam = pygame.camera.Camera(cam_list[0], (CAMERA_WIDTH, CAMERA_HEIGHT))
cam.start()

# Create a PyGame window
screen = pygame.display.set_mode((CAMERA_WIDTH, CAMERA_HEIGHT))
pygame.display.set_caption("Real-time Camera Feed")
font = pygame.font.Font(None, 36)

# Main loop
running = True
last_inference_time = 0  # Last inference time
inference_interval = 1.0  # Infer once per second
display_text = "Waiting for inference..."

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get the camera frame
    image_surface = cam.get_image()
    screen.blit(image_surface, (0, 0))

    # Perform inference (once per second)
    current_time = time.time()
    if current_time - last_inference_time > inference_interval:
        last_inference_time = current_time

        # Preprocess the camera image
        start_time = time.time()
        pixel_values = preprocess_camera_image(image_surface, input_size=448)

        # Send to the model
        question = ('<image>\nObserve the positional relationship between the person\'s hand and the bottle in the image. '
                    'The task is to guide the hand to grab the cup. If the task is not yet complete, please choose one '
                    'of the following commands: forward, backward, left, right, up, down, grasp, release. '
                    'If the task is complete, please output "Task complete". Please output only one of these commands.')
        generation_config = dict(max_new_tokens=128, do_sample=True)
        response = model.chat(tokenizer, pixel_values, question, generation_config)

        # Calculate inference time
        elapsed_time = time.time() - start_time
        print(f"Inference time: {elapsed_time:.3f} seconds")
        print(f"Model output: {response}")

        # Display inference result
        display_text = response

    # Draw the inference text on the screen
    text_surface = font.render(display_text, True, (255, 0, 0))
    screen.blit(text_surface, (10, 10))

    pygame.display.update()

# Release resources
cam.stop()
pygame.quit()