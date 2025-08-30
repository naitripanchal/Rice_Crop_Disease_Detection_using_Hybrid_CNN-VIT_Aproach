import matplotlib.pyplot as plt

# Data
authors = [
    "Hybrid CNN-ViT", 
    "Resnet50", 
    "MobileNetV2", 
    "MobileViT",  
    "DeiT-Tiny"
]

accuracies = [96.54, 93.93, 95.39, 95.45, 94.54]

# Define a unique color for each bar
colors = ['#66c2a5', '#fc8d62', "#8398e0", '#e78ac3', '#a6d854']

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(authors, accuracies, color=colors)

# Add accuracy value ABOVE each bar
for bar, accuracy in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f"{accuracy:.2f}%", 
            ha='center', va='bottom', color='black', fontsize=10, fontweight='bold')

# Axis labels and title
ax.set_ylabel('Accuracy (%)')
ax.set_title('State-of-the-Art Model Comparison')

plt.xticks(rotation=15)
plt.ylim(90, 97)              # ✅ tighter scale, less blank space    # ✅ clean ticks
plt.tight_layout()
plt.savefig('../outputs/state-of-the-art.png', dpi=300)
plt.show()
