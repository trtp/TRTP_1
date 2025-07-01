# import os
# import json
# import torch
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from PIL import Image
#
# import torch
# from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
#
# def load_model(model_path):
#     """
#     加载大模型和处理器。
#     :param model_path: 模型路径。
#     :return: 模型和处理器。
#     """
#
#     model = LlavaOnevisionForConditionalGeneration.from_pretrained(
#         model_path,
#         torch_dtype=torch.float16,
#         low_cpu_mem_usage=True,
#     ).to(0)
#     processor = AutoProcessor.from_pretrained(model_path)
#     return model, processor
#
#
# def infer_task_planning(model, processor, system_value, human_value, video_path):
#     """
#     使用大模型推理视频中的任务规划。
#     :param model: 加载的大模型。
#     :param processor: 模型处理器。
#     :param system_value: system 字段的值。
#     :param human_value: human 字段的值。
#     :param video_path: 视频路径。
#     :return: 推理结果字符串。
#     """
#     # 构造输入消息
#     messages = [
#         {"role": "system", "content": [{"type": "text", "text": system_value}]},
#         {"role": "user", "content": [{"type": "text", "text": "<video>列出视频中的机械手的动作序列，被机械手抓住的物体和动作以及方位要具体"}, {"type": "video", "video": video_path}]},
#     ]
#
#     # 准备推理输入
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to("cuda:0")  # 将输入数据加载到主 GPU
#
#     # 推理
#     with torch.no_grad():
#         generated_ids = model.generate(**inputs, max_new_tokens=512)
#         generated_ids_trimmed = [
#             out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]
#         output_text = processor.batch_decode(
#             generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )[0]
#
#     return output_text
#
#
# def update_dataset_with_inference(model, processor, input_json, output_json):
#     """
#     将大模型推理结果填入数据集的 gpt 字段。
#     :param model: 加载的大模型。
#     :param processor: 模型处理器。
#     :param input_json: 输入 JSON 文件路径。
#     :param output_json: 输出 JSON 文件路径。
#     """
#     # 加载输入数据集
#     with open(input_json, 'r', encoding='utf-8') as f:
#         dataset = json.load(f)
#
#     # 遍历每一项数据，进行推理
#     for item in dataset:
#         system_value = item["conversations"][0]["value"]
#         human_value = item["conversations"][1]["value"]
#         video_path = item["videos"][0]
#
#         # 调用推理函数
#         print(f"正在处理视频: {video_path}")
#         try:
#             gpt_result = infer_task_planning(model, processor, system_value, human_value, video_path)
#             item["conversations"][2]["value"] = gpt_result
#             print(f"推理完成: {gpt_result}")
#         except Exception as e:
#             print(f"推理失败: {video_path}, 错误信息: {e}")
#
#     # 保存更新后的数据集
#     with open(output_json, 'w', encoding='utf-8') as f:
#         json.dump(dataset, f, ensure_ascii=False, indent=4)
#     print(f"更新后的数据集已保存到: {output_json}")
#
#
# if __name__ == "__main__":
#     # 定义新模型路径（如果不同）
#     model_path = "/media/ubuntu/10B4A468B4A451D0/models/llava-onevision-qwen2-7b-ov-hf"
#
#     # 需要批量推理的数据集
#     dataset_pairs = [
#         ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/InternVL2-8B-prompt-output_dataset.json",
#          "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_InternVL2-8B-prompt.json"),
#
#         ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/InternVL2_5-8B-prompt-output_dataset.json",
#          "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_InternVL2_5-8B-prompt.json"),
#
#         ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Llama-3.2-11B-prompt-output_dataset.json",
#          "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_Llama-3.2-11B-_prompt.json"),
#
#         ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/llava-onevision-qwen2-7b-ov-hf-prompt-output_dataset.json",
#          "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_llava-onevision-qwen2-7b-ov-hf-prompt.json"),
#
#         ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/llava-v1.6-vicuna-7b-hf-prompt-output_dataset.json",
#          "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_llava-v1.6-vicuna-7b-hf-prompt.json"),
#
#         ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/MiniCPM-V-2_prompt_6output_dataset.json",
#          "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_MiniCPM-V-2_prompt.json"),
#
#         ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Molmo-7B-D-0924-prompt-output_dataset.json",
#          "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_Molmo-7B-D-0924-prompt.json"),
#
#         ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Ovis1.6-Gemma2-9B_prompt_output_dataset.json",
#          "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_Ovis1.6-Gemma2-9B_prompt.json"),
#
#         ("/home/ubuntu/Desktop/dataset/droidJsonDatset/02/Qwen2VL7B_prompt_dataset.json",
#          "/home/ubuntu/Desktop/dataset/droidJsonDatset/03/llava-onevision-qwen2-7b-ov-hf_infer_Qwen2VL7B_prompt.json")
#     ]
#
#     # 加载模型
#     model, processor = load_model(model_path)
#
#     # 遍历 dataset_pairs 逐个处理
#     for input_json, output_json in dataset_pairs:
#         print(f"Processing: {input_json} -> {output_json}")
#         update_dataset_with_inference(model, processor, input_json, output_json)
#         print(f"Finished processing: {input_json}\n")