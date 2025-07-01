import os
import json
import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
from decord import VideoReader, cpu  # pip install decord

MAX_NUM_FRAMES = 64
params = {}


# 加载模型和处理器
def load_model(model_name):
    """
    加载模型和分词器。
    :param model_name: 模型名称。
    :return: 加载的模型和分词器。
    """
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                      attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


# 编码视频函数
def encode_video(video_path):
    """
    提取视频中的帧。
    :param video_path: 视频路径。
    :return: 帧列表。
    """

    def uniform_sample(l, n):
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
    print('num frames:', len(frames))
    return frames


# 推理函数
def infer_task_planning(model, tokenizer, video_path, question):
    """
    使用模型推理视频中的任务规划。
    :param model: 加载的模型。
    :param tokenizer: 模型分词器。
    :param video_path: 视频路径。
    :param question: 提问。
    :return: 推理结果。
    """
    frames = encode_video(video_path)
    msgs = [
        {'role': 'user', 'content': frames + [question]},
    ]

    # 设置解码参数
    params = {
        "use_image_id": False,
        "max_slice_nums": 2  # 若CUDA OOM且视频分辨率大于448*448，可以设为1
    }

    # 推理并获取结果
    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        **params
    )
    return answer


# 更新数据集函数
def update_dataset_with_inference(model, tokenizer, input_json, output_json):
    """
    将推理结果更新到数据集中。
    :param model: 加载的模型。
    :param tokenizer: 模型分词器。
    :param input_json: 输入数据集路径。
    :param output_json: 输出数据集路径。
    """
    # 加载输入数据集
    with open(input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 遍历每一项数据，进行推理
    for item in dataset:
        video_path = item["videos"][0]
        prompt = item["conversations"][0]["value"]
        #prompt =""
        question = "这是对视频截图空间场景的描述，给你作为参考"  + prompt + "列出视频中的机械手的动作序列"  # 可以修改为实际问题
        #question = "列出视频中的机械手的动作序列，被机械手抓住的物体和动作以及方位要具体"
        # 调用推理函数
        print(f"正在处理视频: {video_path}")
        try:
            gpt_result = infer_task_planning(model, tokenizer, video_path, question)
            item["conversations"][2]["value"] = gpt_result
            print(f"推理完成: {gpt_result}")
        except Exception as e:
            print(f"推理失败: {video_path}, 错误信息: {e}")

    # 保存更新后的数据集
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"更新后的数据集已保存到: {output_json}")


if __name__ == "__main__":
    # 定义路径
    model_path = "/media/ubuntu/10B4A468B4A451D0/models/MiniCPM-V-2_6"

    # 定义 input_json 和 output_json 对应列表
    dataset_pairs = [
        # ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/InternVL2-8B-prompt-output_dataset.json",
        #  "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_InternVL2-8B-prompt.json"),

        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/InternVL2_5-8B-prompt-output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_InternVL2_5-8B-prompt.json"),

        ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Llama-3.2-11B-prompt-output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_Llama-3.2-11B-_prompt.json"),

        # ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/llava-onevision-qwen2-7b-ov-hf-prompt-output_dataset.json",
        #  "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_llava-onevision-qwen2-7b-ov-hf-prompt.json"),
        #
        # ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/llava-v1.6-vicuna-7b-hf-prompt-output_dataset.json",
        #  "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_llava-v1.6-vicuna-7b-hf-prompt.json"),
        #
        # ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/MiniCPM-V-2_prompt_6output_dataset.json",
        #  "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_MiniCPM-V-2_prompt.json"),
        #
        # ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Molmo-7B-D-0924-prompt-output_dataset.json",
        #  "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_Molmo-7B-D-0924-prompt.json"),
        #
        # ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Ovis1.6-Gemma2-9B_prompt_output_dataset.json",
        #  "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_Ovis1.6-Gemma2-9B_prompt.json"),
        #
        # ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Qwen2VL7B_prompt_dataset.json",
        #  "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/MiniCPM-V-2_6_infer_Qwen2VL7B_prompt.json")
    ]

    # 加载模型
    model, tokenizer = load_model(model_path)

    # 遍历 dataset_pairs 逐个处理
    for input_json, output_json in dataset_pairs:
        print(f"Processing: {input_json} -> {output_json}")
        update_dataset_with_inference(model, tokenizer, input_json, output_json)
        print(f"Finished processing: {input_json}\n")

# 使用示例
# if __name__ == "__main__":
#     # 定义路径
#     model_name = "/media/ubuntu/10B4A468B4A451D0/models/MiniCPM-V-2_6"
#     input_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/internvl2_8b_updated_dataset.json"
#     output_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/internvl2_8b_data_mini_cpm_infer_updated_dataset.json"
#
#     # 加载模型
#     model, tokenizer = load_model(model_name)
#
#     # 更新数据集
#     update_dataset_with_inference(model, tokenizer, input_json, output_json)