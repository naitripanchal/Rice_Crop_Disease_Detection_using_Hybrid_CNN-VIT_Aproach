import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Class names
classes = [
    'Brown spot fungal', 'Damaged pests', 'Downy mildew fungal', 'Healthy',
    'Hispa pests', 'Leaf Streak bacterial', 'Leaf blight bacterial', 'Tungro viral'
]

# Counts before augmentation (original)
before_counts = [965, 1442, 620, 1764, 1594, 380, 479, 1088]

# Set all counts after augmentation to 1630
after_counts = [1630] * len(classes)

# Colormap: reversed for light to dark
colors = sns.color_palette("viridis", len(classes))[::-1]

x_positions = np.arange(len(classes)) * 0.7

# Plot side-by-side
fig, axs = plt.subplots(1, 2, figsize=(11, 6))
fig.suptitle("Class Distribution Before vs After Augmentation", fontsize=16)

# BEFORE
axs[0].bar(x_positions, before_counts, color=colors, width=0.5)
axs[0].set_title("Before Augmentation")
axs[0].set_ylabel("Image Count")
for i, count in enumerate(before_counts):
     axs[0].text(x_positions[i], count + 25, str(count), ha='center', va='bottom', fontsize=10)

# AFTER (all 1630)
axs[1].bar(x_positions, after_counts, color=colors, width=0.5)
axs[1].set_title("After Augmentation")
for i, count in enumerate(after_counts):
     axs[1].text(x_positions[i], count + 25, str(count), ha='center', va='bottom', fontsize=10)

# Final touches
for ax in axs:
    ax.set_ylim(0, max(max(before_counts), 1630) + 200)
    ax.set_xticklabels(classes, rotation=45, ha='right')

plt.tight_layout()
plt.savefig("../outputs/class_distribution.png")
plt.show()
