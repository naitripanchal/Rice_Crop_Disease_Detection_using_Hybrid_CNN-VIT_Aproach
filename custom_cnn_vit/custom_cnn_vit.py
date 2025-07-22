# Custom CNN-ViT model training

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ✅ Paths
base_dir = "/kaggle/input/split-dataset/split_dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# ✅ Transforms
transform = {
    "train": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # <= gentle rotation
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ]),
    
    "val": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
}

# ✅ Datasets and Dataloaders
train_dataset = ImageFolder(train_dir, transform=transform["train"])
val_dataset = ImageFolder(val_dir, transform=transform["val"])
test_dataset = ImageFolder(test_dir, transform=transform["val"])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
num_classes = len(train_dataset.classes)


# ✅ Custom CNN-ViT Hybrid Model
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=128, patch_size=16, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=6, dropout=0.05):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 65, embed_dim))  # 64 patches + 1 CLS

    def forward(self, x):
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        return self.encoder(x)

class CNNViTHybrid(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )

        self.patch_embed = PatchEmbed(in_channels=256, patch_size=8, embed_dim=256)
        self.transformer = TransformerEncoder(embed_dim=256, num_heads=4, num_layers=6)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = self.patch_embed(x)
        x = self.transformer(x)
        return self.classifier(x[:, 0])

model = CNNViTHybrid(num_classes=num_classes).to(device)

# ✅ Loss & Optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# ✅ Metric Storage
train_acc_list, train_loss_list, val_acc_list = [], [], []

# ✅ Training Loop
def train_one_epoch(epoch):
    model.train()
    correct, total, running_loss = 0, 0, 0.0
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
    train_acc_list.append(acc)
    train_loss_list.append(running_loss / len(train_loader))
    print(f"Train Acc: {acc:.4f}, Loss: {running_loss / len(train_loader):.4f}")

# ✅ Evaluation Loop
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
    if name == "Validation": val_acc_list.append(acc)
    print(f"{name} Accuracy: {acc:.4f}")
    if name == "Validation":
        return acc  # Return val_acc
    if collect_preds:
        return y_true, y_pred


# ✅ Run Training
for epoch in range(50):
    train_one_epoch(epoch)
    val_acc = evaluate(val_loader)
    scheduler.step(val_acc)

# ✅ Save Model
torch.save(model.state_dict(), "cnn_vit_hybrid.pt")
print("Model saved.")

# ✅ Final Test
y_true, y_pred = evaluate(test_loader, name="Test", collect_preds=True)


# ✅ Accuracy / Loss Plot
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


# ✅ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd',
            norm=mcolors.Normalize(vmin=0, vmax=max(cm.max(), 1)),
            xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes,
            linewidths=0.5, linecolor='gray')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png")


# ✅ Classification Report
report = classification_report(y_true, y_pred, target_names=test_dataset.classes, digits=4)
print("\nClassification Report:\n", report)
with open("classification_report.txt", "w") as f:
    f.write(report)

