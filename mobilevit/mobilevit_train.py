import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from timm import create_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, Normalize

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ✅ Paths
base_dir = "/content/drive/MyDrive/split_dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# ✅ Transforms
transform = {
    "train": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ]),
    "val": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
}

# ✅ Datasets
train_dataset = ImageFolder(train_dir, transform=transform["train"])
val_dataset = ImageFolder(val_dir, transform=transform["val"])
test_dataset = ImageFolder(test_dir, transform=transform["val"])

# ✅ Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# ✅ Model
model = create_model('mobilevit_s', pretrained=True, num_classes=len(train_dataset.classes))
model.to(device)

# ✅ Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ✅ Metrics Storage
train_acc_list = []
train_loss_list = []
val_acc_list = []

# ✅ Train function
def train_one_epoch(epoch):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        running_loss += loss.item()

    acc = correct / total
    avg_loss = running_loss / len(train_loader)
    train_acc_list.append(acc)
    train_loss_list.append(avg_loss)
    print(f"Train Acc: {acc:.4f}, Loss: {avg_loss:.4f}")

# ✅ Eval function
def evaluate(loader, name="Validation", collect_preds=False):
    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"{name} Eval"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            if collect_preds:
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

    acc = correct / total
    if name == "Validation":
        val_acc_list.append(acc)
    print(f"{name} Accuracy: {acc:.4f}")
    if collect_preds:
        return y_true, y_pred

# ✅ Run Training
for epoch in range(25):  # Change this for more epochs
    train_one_epoch(epoch)
    evaluate(val_loader)

# ✅ Save model weights
torch.save(model.state_dict(), "mobilevit_weights.pt")
print("Saved weights to mobilevit_weights.pt")

# ✅ Final Test + Metrics
y_true, y_pred = evaluate(test_loader, name="Test", collect_preds=True)

# ✅ Plot Accuracy and Loss Graph
plt.figure(figsize=(8, 5))
plt.plot(train_acc_list, label='Train Accuracy', marker='o')
plt.plot(val_acc_list, label='Val Accuracy', marker='o')
plt.plot(train_loss_list, label='Train Loss', linestyle='--', marker='x')
plt.title("Training Metrics")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_loss_curve.png")
print("Saved accuracy/loss graph to accuracy_loss_curve.png")

# ✅ Confusion Matrix with Viridis (No Yellow)
colors = [(0.9, 1, 0.9), (0, 0, 0.5)]  # Very Light Green (0.9, 1, 0.9) to Dark Blue (0, 0, 0.5)
custom_cmap = LinearSegmentedColormap.from_list('lightgreen_to_darkblue', colors)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap=custom_cmap,
    norm=mcolors.Normalize(vmin=0, vmax=max(cm.max(), 1)),  # Linear normalization for zeros
    xticklabels=test_dataset.classes,
    yticklabels=test_dataset.classes,
    linewidths=0.5,
    linecolor='gray'
)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion matrix to confusion_matrix.png")


# ✅ Classification Report
report = classification_report(y_true, y_pred, target_names=test_dataset.classes, digits=4)
print("\nClassification Report:\n", report)
with open("classification_report.txt", "w") as f:
    f.write(report)
print("Saved classification report to classification_report.txt")
