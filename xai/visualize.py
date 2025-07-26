import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.colors as mcolors
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==== MODEL DEFINITION ====

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
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 65, embed_dim))  # 64 patches + 1 CLS

    def forward(self, x):
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        return self.encoder(x)

    # ✅ New method to extract attention from last layer only
    def get_last_attention(self, x):
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1)]

        last_layer = self.encoder.layers[-1]

        # 👇 Force raw multi-head attention weights
        attn_output, attn_weights = last_layer.self_attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=False  # <-- KEY FIX
        )

        return attn_weights.detach().cpu()  # shape: [B, heads, tokens, tokens]



class CNNViTHybrid(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
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

# ==== CONFIG ====
model_path = "../custom_cnn_vit/cnn_vit_hybrid.pt"   # Your trained model path
image_path = "2.jpg"                                 # Test image
num_classes = 8                                      # As per your trained model

# ==== LOAD MODEL ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNViTHybrid(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ==== IMAGE PREP ====
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
raw_image = Image.open(image_path).convert("RGB")
input_tensor = transform(raw_image).unsqueeze(0).to(device)
rgb_image = np.array(raw_image.resize((256, 256))) / 255.0

# ==== GRAD-CAM++ ====
target_layer = model.cnn[-3]  # Try -1, -2, or -3 depending on focus
cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
output = model(input_tensor)
predicted_class = output.argmax(dim=1).item()
targets = [ClassifierOutputTarget(predicted_class)]
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
gradcam_image = show_cam_on_image(rgb_image.astype(np.float32), grayscale_cam, use_rgb=True)

# ==== SAVE Grad-CAM OUTPUT ====
cv2.imwrite("../outputs/gradcam_leaf2.png", cv2.cvtColor(gradcam_image, cv2.COLOR_RGB2BGR))
print("✅ Saved Grad-CAM++: ../outputs/gradcam_leaf.png")

# ==== ATTENTION MAP (CLS → Patches) ====
with torch.no_grad():
    x = model.cnn(input_tensor)
    x = model.patch_embed(x)
    attn = model.transformer.get_last_attention(x)  # [B, heads, tokens, tokens]
    print("Attention shape:", attn.shape)


cls_attn = attn[0].mean(0)[0, 1:]  # Mean across heads; CLS token to 64 patch tokens
attn_map = cls_attn.reshape(2, 2).numpy()

# Define bright custom yellow-orange-red colormap
bright_colors = [
    (1.0, 0.9, 0.3),   # softer yellow
    (1.0, 0.4, 0.0),   # warm orange
    (0.8, 0.0, 0.0)    # deep red
]
 # bright red
bright_cmap = mcolors.LinearSegmentedColormap.from_list("bright_ylorred", bright_colors)

plt.figure(figsize=(5, 5))
plt.imshow(attn_map, cmap=bright_cmap)
plt.title("Transformer Attention (CLS to Patches)")
plt.axis("off")
plt.colorbar()
plt.tight_layout()
plt.savefig("../outputs/attention_map2.png")
print("✅ Saved Attention Map: ../outputs/attention_map.png")
