# -*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt
import os


TITLE_FONT_SIZE = 18
LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14
ANNOTATION_FONT_SIZE = 15

# 读取多个JSON文件并提取损失值
def load_losses_from_json(file_paths):
    losses = []
    for path in file_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            losses.append(data['loss'])  # 假设每个JSON中有一个loss字段
    return losses

# 标注框自动防重叠逻辑
def place_annotation(ax, annotations, text, color, x_offset=0.97, y_offset=0.96, step=0.12):
    for _ in range(10):  # 最多尝试10次调整
        overlap = False
        for existing in annotations:
            if abs(existing[0] - x_offset) < 0.05 and abs(existing[1] - y_offset) < 0.05:
                overlap = True
                break
        if not overlap:
            break
        y_offset -= step
        if y_offset < 0.1:
            y_offset = 0.98
            x_offset -= step

    ax.text(
        x_offset, y_offset, text,
        transform=ax.transAxes,
        fontsize=ANNOTATION_FONT_SIZE, color=color,
        ha='right', va='top',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=color, lw=1)
    )
    annotations.append((x_offset, y_offset))
# 标注框按顺序上下排列，避免重叠
def place_annotation1(ax, annotations, text, color, x_offset=0.98, start_y=0.98, step=0.08):
    """
    - annotations: 当前子图已添加的注释框（用于确定当前是第几层）
    - x_offset: 注释框横向位置（默认右上角）
    - start_y: 起始纵坐标（默认顶部）
    - step: 每层的垂直间距
    """
    line_index = len(annotations)  # 当前是第几个框，控制y位置
    y_offset = start_y - line_index * step

    ax.text(
        x_offset, y_offset, text,
        transform=ax.transAxes,
        fontsize=ANNOTATION_FONT_SIZE, color=color,
        ha='right', va='top',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=color, lw=1)
    )
    annotations.append((x_offset, y_offset))

# 绘制损失图并保存为矢量图
def plot_losses_with_comparisons(losses, comparison_losses, labels, comparison_labels, comparison_positions,
                                 custom_annotations, save_path_svg, save_path_pdf):
    # 设置全局 tick 字体大小
    plt.rcParams.update({'font.size': TICK_FONT_SIZE})

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    annotations = [[] for _ in range(6)]

    for i in range(6):
        ax = axes[i // 3, i % 3]
        ax.plot(losses[i], label=labels[i], color='blue')
        annotation = custom_annotations["train"].get(i, labels[i])
        place_annotation(ax, annotations[i], annotation, 'blue')

        ax.set_title(labels[i], fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel('Epoch', fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel('Loss', fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
        ax.grid(True)

    for i, pos_list in comparison_positions.items():
        for pos in pos_list:
            ax = axes[pos // 3, pos % 3]
            ax.plot(comparison_losses[i], label=comparison_labels[i], color='red')
            annotation = custom_annotations["comparison"].get(i, comparison_labels[i])
            place_annotation(ax, annotations[pos], annotation, 'red')

            ax.set_title(f'{labels[pos]}', fontsize=TITLE_FONT_SIZE)
            ax.set_xlabel('Epoch', fontsize=LABEL_FONT_SIZE)
            ax.set_ylabel('Loss', fontsize=LABEL_FONT_SIZE)
            ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
            ax.grid(True)

    plt.tight_layout()
    print(f"Saving SVG: {save_path_svg}")
    plt.savefig(save_path_svg, format='svg')
    print(f"Saving PDF: {save_path_pdf}")
    plt.savefig(save_path_pdf, format='pdf')
    plt.show()


# ✅ 模型和对比 JSON 文件路径
train_loss_paths = [
    'C:/Users/quyuanjin/Desktop/研究相关/VLP loss图/InternVL2_5-8B_infer_InternVL2-8B-promptloss_data.json',
    'C:/Users/quyuanjin/Desktop/研究相关/VLP loss图/InternVL2-8B_infer_llava-onevision-qwen2-7b-ov-hf-promptloss_data.json',
    'C:/Users/quyuanjin/Desktop/研究相关/VLP loss图/InternVL2-8B_infer_Ovis1.6-Gemma2-9B_promptloss_data.json',
    'C:/Users/quyuanjin/Desktop/研究相关/VLP loss图/InternVL2-8B_infer_Qwen2VL7B_promptloss_data.json',
    'C:/Users/quyuanjin/Desktop/研究相关/VLP loss图/MiniCPM-V-2_6_infer_InternVL2_5-8B-promptloss_data.json',
    'C:/Users/quyuanjin/Desktop/研究相关/VLP loss图/MiniCPM-V-2_6_infer_Llama-3.2-11B-_promptloss_data.json'
]

comparison_loss_paths = [
    'C:/Users/quyuanjin/Desktop/研究相关/VLP loss图/InternVL2_5-8B_inferloss_data.json',
    'C:/Users/quyuanjin/Desktop/研究相关/VLP loss图/InternVL2-8B_inferloss_data.json',
    'C:/Users/quyuanjin/Desktop/研究相关/VLP loss图/MiniCPM-V-2_6_inferloss_data.json'
]

custom_annotations = {
    "train": {
        0: "InternVL2-8B as prompt",
        1: "llava-onevision-qwen2-7b-ov-hf as prompt",
        2: "Ovis1.6-Gemma2-9B as prompt",
        3: "Qwen2VL7B as prompt",
        4: "InternVL2_5-8B as prompt",
        5: "Llama-3.2-11B as prompt"
    },
    "comparison": {
        0: "without prompt",
        1: "without prompt",
        2: "without prompt"
    }
}

labels = ['InternVL2_5-8B', 'InternVL2-8B', 'InternVL2-8B', 'InternVL2-8B', 'MiniCPM-V-2_6', 'MiniCPM-V-2_6']
comparison_labels = ['InternVL2_5', 'InternVL2', 'MiniCPM-V-2_6']

train_losses = load_losses_from_json(train_loss_paths)
comparison_losses = load_losses_from_json(comparison_loss_paths)

comparison_positions = {
    0: [0],
    1: [1, 2, 3],
    2: [4, 5]
}

output_folder = "C:/Users/quyuanjin/Desktop/研究相关/VLP loss图"
svg_path = os.path.join(output_folder, "loss_comparison_plot.svg")
pdf_path = os.path.join(output_folder, "loss_comparison_plot.pdf")

plot_losses_with_comparisons(
    train_losses, comparison_losses, labels, comparison_labels,
    comparison_positions, custom_annotations, svg_path, pdf_path
)
