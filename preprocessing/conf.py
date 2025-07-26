import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Class labels
labels = [
    'Brown spot fungal', 'Damaged pests', 'Downy mildew fungal', 'Healthy',
    'Hispa pests', 'Leaf Streak bacterial', 'Leaf blight bacterial', 'Tungro viral'
]

# Updated confusion matrix values from image
conf_matrix = np.array([
    [177, 1, 1, 1, 3, 0, 3, 5],
    [  1, 232, 0, 1, 1, 0, 2, 1],
    [  2, 0, 163, 3, 1, 0, 3, 2],
    [  1, 0, 2, 257, 3, 2, 0, 1],
    [  2, 0, 0, 1, 234, 1, 1, 1],
    [  0, 0, 0, 0, 2, 153, 1, 0],
    [  3, 1, 0, 2, 1, 3, 150, 2],
    [  1, 0, 4, 1, 3, 2, 1, 206]
])
# Plot
plt.figure(figsize=(8, 8))
sns.heatmap(
    conf_matrix, annot=True, fmt='d', cmap='OrRd',
    xticklabels=labels, yticklabels=labels,
    square=True, linewidths=0.3, linecolor='gray',
    cbar_kws={"aspect": 20, "shrink": 0.7}
)

# Axis and labels
plt.title("Confusion Matrix - MobileViT")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Ensure layout is applied BEFORE saving
plt.tight_layout()
# plt.savefig("cnn_vit_confusion_updated.png")
plt.show()
