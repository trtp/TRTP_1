import cv2
from PIL import Image
import time
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

torch.cuda.empty_cache()

# 1. 加载模型和预处理器
model_path = "/home/ubuntu/Desktop/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    ignore_mismatched_sizes=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
processor = AutoProcessor.from_pretrained(model_path)


# 2. 定义控制机械臂的函数（实际使用时需调用具体机械臂接口）
def send_command_to_robotic_arm(command):
    print("发送给机械臂的指令：", command)


# 3. 封装模型推理部分为函数
def model_inference(messages, processor, model, device):
    """
    对输入的消息进行处理，并使用模型进行推理，返回生成的文本结果。
    """
    # 生成文本提示和处理视觉信息
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # 模型推理
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


# 4. 定义允许的机械臂指令（增加“任务完成”作为退出指令）
allowed_commands = ["往前", "往后", "往左", "往右", "往上", "往下", "待机", "任务完成"]

# 5. 初始化摄像头并显示界面
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # 如有需要可调整索引或后端
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法从摄像头读取画面")
            break

        # 将 OpenCV BGR 图像转换为 RGB，并转换为 PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # 构造输入消息：当任务完成时输出“任务完成”，否则仅输出六个方向之一
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请观察图像中人的手和瓶子的位置关系。任务是指挥手拿到杯子。若任务尚未完成，请选择以下指令之一：往前，往后，往左，往右，往上，往下，抓住，松开。若任务完成了，请输出“任务完成”。请只输出其中一个命令。"

                        )
                    },
                    {"type": "image", "image": pil_image},
                ],
            }
        ]

        start_time = time.time()  # 开始计时

        # 调用模型推理函数
        output_text = model_inference(messages, processor, model, device)
        elapsed_time = time.time() - start_time

        print(f"本次推理耗时: {elapsed_time:.6f} 秒")

        # 获取模型输出，并做简单过滤
        command = output_text[0].strip() if output_text else ""
        if command not in allowed_commands:
            command = "待机"
        print("模型输出：", command)

        # 下发指令给机械臂
        send_command_to_robotic_arm(command)

        # 如果模型输出“任务完成”，则退出循环
        if command == "任务完成":
            print("任务完成，停止推理。")
            break

        # 在窗口中显示摄像头画面，按 'q' 键也可手动退出
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()