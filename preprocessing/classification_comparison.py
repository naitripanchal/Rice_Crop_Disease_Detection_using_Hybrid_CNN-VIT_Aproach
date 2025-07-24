import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Model names
models = [
    "Proposed Methodology",
    "MobileViT",
    "MobileNetV2",
    "DeiT-Tiny",
    "ResNet50"
]

# Metric values
accuracy = [0.9654, 0.9545, 0.9539, 0.9454, 0.9393]
precision = [0.9654, 0.9548, 0.9538, 0.9467, 0.9401]
recall = [0.9654, 0.9545, 0.9539, 0.9454, 0.9393]
f1_score = [0.9653, 0.9544, 0.9537, 0.9454, 0.9391]

# Bar setup
bar_width = 0.2
x = np.arange(len(models))
colors = {
    'Accuracy': '#1f77b4',   # Blue
    'Precision': '#ffcc00',  # Yellow
    'Recall': '#2ca02c',     # Green
    'F1 Score': '#ff7f0e'    # Orange
}

plt.figure(figsize=(13, 7))
bars1 = plt.bar(x - 1.5*bar_width, accuracy, width=bar_width, color=colors['Accuracy'])
bars2 = plt.bar(x - 0.5*bar_width, precision, width=bar_width, color=colors['Precision'])
bars3 = plt.bar(x + 0.5*bar_width, recall, width=bar_width, color=colors['Recall'])
bars4 = plt.bar(x + 1.5*bar_width, f1_score, width=bar_width, color=colors['F1 Score'])

# Label inside bars (vertical)
def label_vertical(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height - 0.003, f'{height:.4f}',
                 ha='center', va='top', fontsize=8, rotation=90, color='black')

label_vertical(bars1)
label_vertical(bars2)
label_vertical(bars3)
label_vertical(bars4)

# Label above each blue (accuracy) bar, centered
# Add accuracy label directly above each blue bar, centered
for bar in bars1:
    height = bar.get_height()
    x_pos = bar.get_x() + 2 * bar.get_width()
    plt.text(x_pos, height + 0.0035, f'Accuracy = {height * 100} %',
             ha='center', va='bottom', fontsize=9, fontweight='bold')


# Axes
plt.xticks(x, models, rotation=30, ha='right')
plt.ylim(0.90, 0.98)
plt.ylabel("Score")
plt.title("Model Performance Comparison")

# Legend below x-axis
legend_elements = [Patch(facecolor=color, label=label) for label, color in colors.items()]
plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
           fancybox=True, shadow=False, ncol=4)

plt.tight_layout()
plt.savefig("../outputs/model_comparison.png", bbox_inches='tight')
plt.show()
