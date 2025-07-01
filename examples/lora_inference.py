import yaml
import json
from loguru import logger
import time
import sys
from src.llamafactory.chat import ChatModel

if __name__ == '__main__':
    with open('../examples/qyj/qwen_sft_predict.yaml', 'r', encoding='utf-8') as f:
        param = yaml.safe_load(f)

    chat_model = ChatModel(param)

    with open('E:/code/DataSets/qwentest/train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # pre hot
    messages = [{"role": "user", "content": data[0]['instruction']}]
    _ = chat_model.chat(messages)
    print(_)
    # predict_1000 = []
    # total_time = 0
    # for i, item in enumerate(data):
    #     messages = [{"role": "user", "content": item['instruction']}]