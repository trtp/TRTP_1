# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llamafactory.train.tuner import run_exp
import yaml


def main(yaml_path_):
    with open(yaml_path_, 'r', encoding='utf-8') as f:
        param = yaml.safe_load(f)
    run_exp(param)


if __name__ == "__main__":
    # 1.sft指令微调
    # yaml_path = '../examples/qyj/qwen2_lora_sft.yaml'
    # 2.奖励模型训练
    # yaml_path = '../examples/qyj/qwen2_lora_reward.yaml'
    # 3.rlhf-ppo训练
    # yaml_path = '../examples/qyj/qwen2_lora_ppo.yaml'


    # 1.sft指令微调
    # yaml_path = '../examples/qyj_vl/qwen2VL_lora_sft.yaml'
    #yaml_path = '../examples/train_lora/qwen2vl_lora_sft.yaml'
    # 2.奖励模型训练
    # yaml_path = '../examples/qyj_vl/qwen2VL_lora_reward.yaml'
    # 3.rlhf-ppo训练
    yaml_path = '../examples/qyj_vl/qwen2VL_lora_ppo.yaml'

    main(yaml_path)