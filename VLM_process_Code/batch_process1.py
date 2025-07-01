import os
from VLM_process_Code.makeDatasets.Llama32_11BVis_Ins.Llama32 import process_image
from VLM_process_Code.makeDatasets.InternVL2.InternVL2Function import InternVL2_process_image
# 批量处理函数
def batch_process(image_path, new_message_text, output_dir="./outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # 假设需要处理的多个文件及对应文本内容
    files_to_process = [
        {"image_path": image_path, "message_text": new_message_text}
    ]

    outputs = []
    for idx, file_info in enumerate(files_to_process):
        try:
            # 调用封装的函数处理单个任务
            result = process_image(file_info["image_path"], file_info["message_text"])
            outputs.append({"file_idx": idx, "output": result})

            result1 = InternVL2_process_image(file_info["image_path"], file_info["message_text"])
            outputs.append({"file_idx": idx, "output": result1})

            # 保存结果到输出目录
            output_file_path = os.path.join(output_dir, f"result_{idx}.txt")
            with open(output_file_path, "w") as f:
                f.write(result)
            print(f"Processed {file_info['image_path']} successfully.")
        except Exception as e:
            print(f"Error processing file {file_info['image_path']}: {e}")

    return outputs




# 示例调用
if __name__ == "__main__":
    # 图片路径和批量更新的消息文本
    image_path = "/home/ubuntu/Desktop/LLaMA-Factory/assets/4.jpg"
    new_message_text = "This is the updated haiku text for batch processing."

    # 调用批量处理函数
    results = batch_process(image_path, new_message_text)
    for res in results:
        print(f"File Index: {res['file_idx']}, Output: {res['output']}")