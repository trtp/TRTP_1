import cv2
from PIL import Image
import time
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

torch.cuda.empty_cache()

# 1. Load the model and processor
model_path = "/home/ubuntu/Desktop/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    ignore_mismatched_sizes=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
processor = AutoProcessor.from_pretrained(model_path)


# 2. Define a function to control the robotic arm (in a real scenario, this would call a specific arm's API)
def send_command_to_robotic_arm(command):
    print("Sending command to robotic arm:", command)


# 3. Encapsulate the model inference part into a function
def model_inference(messages, processor, model, device):
    """
    Processes the input messages and performs inference with the model,
    returning the generated text result.
    """
    # Generate text prompt and process visual information
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

    # Model inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


# 4. Define allowed robotic arm commands (add "Task complete" as an exit command)
allowed_commands = ["forward", "backward", "left", "right", "up", "down", "standby", "Task complete"]

# 5. Initialize the camera and display the feed
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Adjust index or backend if necessary
if not cap.isOpened():
    print("Could not open camera")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame from camera")
            break

        # Convert OpenCV BGR image to RGB, then to a PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Construct the input message
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Observe the positional relationship between the person's hand and the bottle in the image. "
                            "The task is to guide the hand to grab the cup. If the task is not yet complete, please "
                            "choose one of the following commands: forward, backward, left, right, up, down, grasp, release. "
                            "If the task is complete, please output 'Task complete'. "
                            "Please output only one of these commands."
                        )
                    },
                    {"type": "image", "image": pil_image},
                ],
            }
        ]

        start_time = time.time()  # Start timing

        # Call the model inference function
        output_text = model_inference(messages, processor, model, device)
        elapsed_time = time.time() - start_time

        print(f"Inference time for this cycle: {elapsed_time:.6f} seconds")

        # Get the model output and apply simple filtering
        command = output_text[0].strip() if output_text else ""
        if command not in allowed_commands:
            command = "standby"
        print("Model output:", command)

        # Send the command to the robotic arm
        send_command_to_robotic_arm(command)

        # If the model outputs "Task complete", exit the loop
        if command == "Task complete":
            print("Task complete, stopping inference.")
            break

        # Display the camera feed in a window, press 'q' to exit manually
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()