import pygame
import pygame.camera
import time
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 初始化 PyGame 和摄像头
pygame.init()
pygame.camera.init()

# 获取可用摄像头
cam_list = pygame.camera.list_cameras()
if not cam_list:
    print("未找到摄像头")
    exit()

# 选择第一个摄像头
cam = pygame.camera.Camera(cam_list[0], (640, 480))
cam.start()

# 创建窗口
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("摄像头实时画面")

# 加载模型和预处理器
model_path = "/home/ubuntu/Desktop/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, ignore_mismatched_sizes=True
).to("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(model_path)


# 定义控制机械臂的函数
def send_command_to_robotic_arm(command):
    print("发送给机械臂的指令：", command)


# 允许的机械臂指令
allowed_commands = ["往前", "往后", "往左", "往右", "往上", "往下", "待机", "任务完成"]


# 推理函数
def model_inference(image):
    start_time = time.time()  # 记录推理开始时间

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "请观察图像中人的手和瓶子的位置关系。任务是指挥手拿到杯子。若任务尚未完成，请选择以下指令之一：往前，往后，往左，往右，往上，往下，抓住，松开。若手拿到杯子了，请输出“任务完成”。请只输出其中一个命令。"
                        # "描述图片中的内容。"

                    ),
                },
                {"type": "image", "image": image},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # 模型推理
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # 记录推理时间
    elapsed_time = time.time() - start_time
    print(f"推理时间: {elapsed_time:.6f} 秒")
    print(f"模型输出: {output_text}")

    return output_text if output_text in allowed_commands else "待机"


# 主循环
running = True
last_inference_time = 0  # 记录上次推理时间
inference_interval = 0.1  # 每隔1秒推理一次
command_display = "待机"  # 记录当前的指令

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 获取摄像头画面
    image_surface = cam.get_image()
    screen.blit(image_surface, (0, 0))

    # 每隔一段时间进行推理
    current_time = time.time()
    if current_time - last_inference_time > inference_interval:
        last_inference_time = current_time

        # 将 pygame.Surface 转换为 PIL.Image
        image_rgb = pygame.surfarray.array3d(image_surface)  # 获取 RGB 数组
        image_pil = Image.fromarray(image_rgb.transpose(1, 0, 2))  # 转换为 PIL.Image

        # 调用模型
        command_display = model_inference(image_pil)

        # 发送指令
        send_command_to_robotic_arm(command_display)

        # 如果任务完成，则退出循环
        if command_display == "任务完成":
            print("任务完成，停止推理。")
            running = False

    # 在屏幕上显示当前指令
    font = pygame.font.Font(None, 36)
    text_surface = font.render(f"指令: {command_display}", True, (255, 0, 0))
    screen.blit(text_surface, (10, 10))

    pygame.display.update()

# 释放资源
cam.stop()
pygame.quit()
