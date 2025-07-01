# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


models = [
    "Qwen2VL 7B", "InternVL2 8B", "InternVL2.5", "LLama3.2 11B",
    "MiniCPM V 2.6", "Molmo", "Ovis1.6", "Llava qwen2 OV"
]


data = np.array([
    [0.89, 0.85, 0.80, 0.78, 0.82, 0.79, 0.75, 0.77],
    [0.85, 0.90, 0.88, 0.83, 0.81, 0.76, 0.74, 0.78],
    [0.80, 0.88, 0.92, 0.86, 0.84, 0.77, 0.72, 0.79],
    [0.78, 0.83, 0.86, 0.87, 0.82, 0.74, 0.80, 0.76],
    [0.82, 0.81, 0.84, 0.82, 0.86, 0.79, 0.73, 0.80],
    [0.79, 0.76, 0.77, 0.74, 0.79, 0.81, 0.71, 0.75],
    [0.75, 0.74, 0.72, 0.80, 0.73, 0.71, 0.79, 0.68],
    [0.77, 0.78, 0.79, 0.76, 0.80, 0.75, 0.68, 0.85]
])

# DataFrame
df = pd.DataFrame(data, columns=models, index=models)

# plot
sns.set(style='white')
fig, ax = plt.subplots(figsize=(10, 8))


sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, linewidths=0.5, ax=ax)


ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xticklabels([label.replace(' ', '\n') for label in models], rotation=0)


diagonal_color = 'gray'
for i in range(len(models)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color=diagonal_color, alpha=1))

legend_elements = [Patch(facecolor=diagonal_color, edgecolor='gray', label='Diagonal (Excluded)')]
ax.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.825, 0),
    title="Legend",
    ncol=1
)

# plt.title("Cross Validation Scores Heatmap (Diagonal Highlighted)", fontsize=16, pad=20)
# plt.xlabel('Model', fontsize=12)
# plt.ylabel('Model', fontsize=12)


# plt.tight_layout()
# plt.show()

plt.savefig("CrossValHeatMap.pdf", format="pdf", bbox_inches="tight")
# SVG
plt.savefig("CrossValHeatMap.svg", format="svg", bbox_inches="tight")
# PNG
plt.savefig("CrossValHeatMap.png", format="png", bbox_inches="tight")


plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()
