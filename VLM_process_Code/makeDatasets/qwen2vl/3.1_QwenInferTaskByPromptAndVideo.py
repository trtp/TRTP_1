import os
import json
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm

def load_model(model_path):
    """
    加载大模型和处理器。
    :param model_path: 模型路径。
    :return: 模型和处理器。
    """
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        offload_folder="offload",
        offload_state_dict=True,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def infer_task_planning(model, processor, system_value, human_value, video_path):
    """
    使用大模型推理视频中的任务规划。
    :param model: 加载的大模型。
    :param processor: 模型处理器。
    :param system_value: system 字段的值。
    :param human_value: human 字段的值。
    :param video_path: 视频路径。
    :return: 推理结果字符串。
    """
    # 构造输入消息
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_value}]},
      #  {"role": "user", "content": [{"type": "text", "text": human_value}, {"type": "video", "video": video_path}]},
        {"role": "user", "content": [{"type": "text", "text": "<video>列出视频中的机械手的动作序列"}, {"type": "video","fps": 1.0, "video": video_path}]},
    ]

    # 准备推理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda:0")  # 将输入数据加载到主 GPU

    # 推理
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return output_text


def update_dataset_with_inference(model, processor, input_json, output_json):
    """
    将大模型推理结果填入数据集的 gpt 字段。
    :param model: 加载的大模型。
    :param processor: 模型处理器。
    :param input_json: 输入 JSON 文件路径。
    :param output_json: 输出 JSON 文件路径。
    """
    # 加载输入数据集
    with open(input_json, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 遍历每一项数据，进行推理
    for item in dataset:
        system_value = item["conversations"][0]["value"]
        human_value = item["conversations"][1]["value"]
        video_path = item["videos"][0]

        # 调用推理函数
        print(f"正在处理视频: {video_path}")
        try:
            gpt_result = infer_task_planning(model, processor, system_value, human_value, video_path)
            item["conversations"][2]["value"] = gpt_result
            print(f"推理完成: {gpt_result}")
        except Exception as e:
            print(f"推理失败: {video_path}, 错误信息: {e}")

    # 保存更新后的数据集
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"更新后的数据集已保存到: {output_json}")


def get_model_name(model_path):
    """从模型路径中提取模型名称"""
    return os.path.basename(model_path)


if __name__ == "__main__":
    # 模型路径列表
    model_paths = [
        #"/home/ubuntu/Desktop/Qwen2-VL-7B-Instruct",
        #"/home/ubuntu/Desktop/LLaMA-Factory/src/saves/Qwen2-VL-7B-Instruct/lora/merge/InternVL2-8B_infer",
        #"/home/ubuntu/Desktop/LLaMA-Factory/src/saves/Qwen2-VL-7B-Instruct/lora/merge/InternVL2-8B_infer_llava-onevision-qwen2-7b-ov-hf-prompt",
        #"/home/ubuntu/Desktop/LLaMA-Factory/src/saves/Qwen2-VL-7B-Instruct/lora/merge/InternVL2-8B_infer_Ovis1.6-Gemma2-9B_prompt",
        #"/home/ubuntu/Desktop/LLaMA-Factory/src/saves/Qwen2-VL-7B-Instruct/lora/merge/InternVL2-8B_infer_Qwen2VL7B_prompt",
        #"/home/ubuntu/Desktop/LLaMA-Factory/src/saves/Qwen2-VL-7B-Instruct/lora/merge/InternVL2_5-8B_infer",
        #"/home/ubuntu/Desktop/LLaMA-Factory/src/saves/Qwen2-VL-7B-Instruct/lora/merge/InternVL2_5-8B_infer_InternVL2-8B-prompt",
        "/home/ubuntu/Desktop/LLaMA-Factory/src/saves/Qwen2-VL-7B-Instruct/lora/merge/MiniCPM-V-2_6_infer",
        #"/home/ubuntu/Desktop/LLaMA-Factory/src/saves/Qwen2-VL-7B-Instruct/lora/merge/MiniCPM-V-2_6_infer_InternVL2_5-8B-prompt",
        #"/home/ubuntu/Desktop/LLaMA-Factory/src/saves/Qwen2-VL-7B-Instruct/lora/merge/MiniCPM-V-2_6_infer_Llama-3.2-11B-_prompt"
    ]

    # 输入与输出 JSON 数据集
    dataset_pairs = [
        ("/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/Ovis1.6-Gemma2-9B_prompt_output_dataset.json",
         "/home/ubuntu/Desktop/dataset/droidJsonDatsetTest/output")  # 输出文件夹路径
    ]

    # 遍历数据集与模型
    for input_json, output_folder in dataset_pairs:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print(f"Processing dataset: {input_json}")

        for model_path in tqdm(model_paths, desc="Model Inference"):
            model_name = get_model_name(model_path)

            # 输出文件名格式：模型名称_qwen2vlsft.json
            output_json = os.path.join(output_folder, f"{model_name}_qwen2vlsft.json")

            print(f"\nUsing model: {model_name}")

            # 加载模型
            model, tokenizer = load_model(model_path)

            # 执行推理并保存输出
            update_dataset_with_inference(model, tokenizer, input_json, output_json)

            print(f"Finished processing with {model_name}: {output_json}")

    print("\n✅ All models have finished processing.")

# 使用示例
# if __name__ == "__main__":
#     # 定义路径
#     model_path = "/home/ubuntu/Desktop/Qwen2-VL-7B-Instruct"
#     # model_path = "/saves/Qwen2-VL-2B-Instruct/lora/sftmerge"
#     input_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/internvl2_8b_updated_dataset.json"
#     output_json = "/home/ubuntu/Desktop/dataset/droidJsonDatset/internvl2_8b_data_qwenvl_7b_infer_updated_dataset_updated_dataset.json"
#
#     # 加载模型
#     model, processor = load_model(model_path)
#
#     # 更新数据集
#     update_dataset_with_inference(model, processor, input_json, output_json)