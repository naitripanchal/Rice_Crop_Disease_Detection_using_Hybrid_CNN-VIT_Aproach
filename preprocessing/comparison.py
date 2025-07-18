# COMAPRISON OF BALANACED AND IMBALANCED DATASET

import os
import matplotlib.pyplot as plt
import numpy as np

original_path = "../dataset/train_images"
balanced_path = "../dataset/train_images_balanced"

def count_images(folder_path):
    class_counts = {}
    for class_name in os.listdir(folder_path):
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            count = len([
                f for f in os.listdir(class_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            class_counts[class_name] = count
    return class_counts

original_counts = count_images(original_path)
balanced_counts = count_images(balanced_path)

class_names = sorted(original_counts.keys())
x = np.arange(len(class_names))

viridis = plt.cm.get_cmap('viridis', len(class_names))
colors = [viridis(i) for i in reversed(range(len(class_names)))]

fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

bars1 = axs[0].bar(x, [original_counts[c] for c in class_names], color=colors)
axs[0].set_title("Before Augmentation")
axs[0].set_ylabel("Image Count")
axs[0].set_xticks(x)
axs[0].set_xticklabels(class_names, rotation=45, ha='right')
axs[0].bar_label(bars1, padding=3)

bars2 = axs[1].bar(x, [balanced_counts[c] for c in class_names], color=colors)
axs[1].set_title("After Augmentation")
axs[1].set_xticks(x)
axs[1].set_xticklabels(class_names, rotation=45, ha='right')
axs[1].bar_label(bars2, padding=3)

plt.suptitle("Class Distribution Before vs After Augmentation", fontsize=16)
plt.tight_layout()
plt.savefig("../outputs/class_distribution_combined.png")
plt.close()

# --- BEFORE AUGMENTATION ONLY ---
fig, ax1 = plt.subplots(figsize=(8, 8))
bars1 = ax1.bar(x, [original_counts[c] for c in class_names], color=colors)
ax1.set_title("Before Augmentation")
ax1.set_ylabel("Image Count")
ax1.set_xticks(x)
ax1.set_xticklabels(class_names, rotation=45, ha='right')
ax1.bar_label(bars1, padding=3)
plt.tight_layout()
plt.savefig("../outputs/class_distribution_before.png")
plt.close()

# --- AFTER AUGMENTATION ONLY ---
fig, ax2 = plt.subplots(figsize=(8, 8))
bars2 = ax2.bar(x, [balanced_counts[c] for c in class_names], color=colors)
ax2.set_title("After Augmentation")
ax2.set_ylabel("Image Count")
ax2.set_xticks(x)
ax2.set_xticklabels(class_names, rotation=45, ha='right')
ax2.bar_label(bars2, padding=3)
plt.tight_layout()
plt.savefig("../outputs/class_distribution_after.png")
plt.close()
