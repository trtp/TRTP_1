import numpy as np
import torch
import torchvision.transforms as T
import pygame
import pygame.camera
import time
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from modelscope import AutoModel, AutoTokenizer

# 设定常量
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 摄像头分辨率
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
#INTERNVL_MODEL_PATH = '/media/ubuntu/10B4A468B4A451D0/models/InternVL2_5-8B'
INTERNVL_MODEL_PATH = '/media/ubuntu/10B4A468B4A451D0/models/InternVL2-8B'
# 预处理
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

# 动态预处理
def dynamic_preprocess(image, image_size=448, max_num=12):
    orig_width, orig_height = image.size
    resized_img = image.resize((image_size, image_size))
    return [resized_img]  # 这里简化成单张图像，不进行切块

# 处理摄像头图像
def preprocess_camera_image(image_surface, input_size=448):
    image_rgb = pygame.surfarray.array3d(image_surface)  # 转换为 numpy 数组
    image_pil = Image.fromarray(image_rgb.transpose(1, 0, 2))  # 转换为 PIL.Image
    processed_images = dynamic_preprocess(image_pil, image_size=input_size)
    transform = build_transform(input_size=input_size)
    pixel_values = [transform(image) for image in processed_images]
    return torch.stack(pixel_values).to(torch.bfloat16).cuda()

# 加载 InternVL2.5-8B
print("加载模型...")
model = AutoModel.from_pretrained(
    INTERNVL_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(INTERNVL_MODEL_PATH, trust_remote_code=True, use_fast=False)
print("模型加载完成！")

# 初始化摄像头
pygame.init()
pygame.camera.init()
cam_list = pygame.camera.list_cameras()
if not cam_list:
    print("未找到摄像头")
    exit()
cam = pygame.camera.Camera(cam_list[0], (CAMERA_WIDTH, CAMERA_HEIGHT))
cam.start()

# 创建 PyGame 窗口
screen = pygame.display.set_mode((CAMERA_WIDTH, CAMERA_HEIGHT))
pygame.display.set_caption("摄像头实时画面")
font = pygame.font.Font(None, 36)

# 主循环
running = True
last_inference_time = 0  # 上次推理时间
inference_interval = 1.0  # 每秒推理一次
display_text = "等待推理..."

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 获取摄像头画面
    image_surface = cam.get_image()
    screen.blit(image_surface, (0, 0))

    # 进行推理（每秒一次）
    current_time = time.time()
    if current_time - last_inference_time > inference_interval:
        last_inference_time = current_time

        # 预处理摄像头图像
        start_time = time.time()
        pixel_values = preprocess_camera_image(image_surface, input_size=448)

        # 发送给模型
        question = '<image>\n请观察图像中人的手和瓶子的位置关系。任务是指挥手拿到杯子。若任务尚未完成，请选择以下指令之一：往前，往后，往左，往右，往上，往下，抓住，松开。若任务完成了，请输出“任务完成”。请只输出其中一个命令。'
        generation_config = dict(max_new_tokens=128, do_sample=True)
        response = model.chat(tokenizer, pixel_values, question, generation_config)

        # 计算推理时间
        elapsed_time = time.time() - start_time
        print(f"推理时间: {elapsed_time:.3f} 秒")
        print(f"模型输出: {response}")

        # 显示推理结果
        display_text = response

    # 在屏幕上绘制推理文本
    text_surface = font.render(display_text, True, (255, 0, 0))
    screen.blit(text_surface, (10, 10))

    pygame.display.update()

# 释放资源
cam.stop()
pygame.quit()