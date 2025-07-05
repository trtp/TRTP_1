import os
import json
import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
from decord import VideoReader, cpu  # pip install decord

MAX_NUM_FRAMES = 64
params = {}


# Load the model and processor
def load_model(model_name):
    """
    Loads the model and its tokenizer.
    :param model_name: The name of the model.
    :return: The loaded model and tokenizer.
    """
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                      attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


# Video encoding function
def encode_video(video_path):
    """
    Extracts frames from the video.
    :param video_path: Path to the video.
    :return: A list of frames.
    """

    def uniform_sample(l, n):
        if len(l) <= n:
            return l
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('Number of frames:', len(frames))
    return frames


# Inference function
def infer_task_planning(model, tokenizer, video_path, question):
    """
    Infers the task plan from a video using the model.
    :param model: The loaded model.
    :param tokenizer: The model's tokenizer.
    :param video_path: Path to the video.
    :param question: The question or prompt.
    :return: The inference result.
    """
    frames = encode_video(video_path)
    msgs = [
        {'role': 'user', 'content': frames + [question]},
    ]

    # Set decoding parameters
    params = {
        "use_image_id": False,
        "max_slice_nums": 2  # If CUDA OOM and video resolution is > 448x448, this can be set to 1
    }

    # Infer and get the result
    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        **params
    )
    return answer


# Dataset update function
def update_dataset_with_inference(model, tokenizer, input_json, output_json):
    """
    Updates the dataset with the inference results.
    :param model: The loaded model.
    :param tokenizer: The model's tokenizer.
    :param input_json: Path to the input dataset.
    :param output_json: Path to the output dataset.
    """
    # Load the input dataset
    with open(input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Iterate through each item and perform inference
    for item in dataset:
        video_path = item["videos"][0]
        prompt = item["conversations"][0]["value"]
        question = "This is a description of the spatial scene from the video frames, for your reference:" + prompt + "List the action sequence of the robotic arm in the video"

        # Call the inference function
        print(f"Processing video: {video_path}")
        try:
            gpt_result = infer_task_planning(model, tokenizer, video_path, question)
            item["conversations"][2]["value"] = gpt_result
            print(f"Inference complete: {gpt_result}")
        except Exception as e:
            print(f"Inference failed for: {video_path}, Error: {e}")

    # Save the updated dataset
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"Updated dataset has been saved to: {output_json}")


if __name__ == "__main__":
    # Define paths
    model_path = "/media/ubuntu/10B4A468B4A451D0/models/MiniCPM-V-2_6"

    # Define a list of corresponding input_json and output_json pairs
    dataset_pairs = [
        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/InternVL2_5-8B-prompt-output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_InternVL2_5-8B-prompt.json"),
        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Llama-3.2-11B-prompt-output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_Llama-3.2-11B-_prompt.json"),
    ]

    # Load the model
    model, tokenizer = load_model(model_path)

    # Iterate through dataset_pairs and process each one
    for input_json, output_json in dataset_pairs:
        print(f"Processing: {input_json} -> {output_json}")
        update_dataset_with_inference(model, tokenizer, input_json, output_json)
        print(f"Finished processing: {input_json}\n")