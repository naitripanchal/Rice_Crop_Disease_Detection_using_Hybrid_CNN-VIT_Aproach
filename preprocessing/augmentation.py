# AUGMENT DATA I.E. BALANCE DATASET FOR MODEL ACCURACY

import os
import random
from PIL import Image
from torchvision import transforms
import shutil

source_dir = "../dataset/train_images"
target_dir = "../dataset/train_images_balanced"

target_counts = {
    "Healthy": 1764,               
    "Hispa pests": 1628,
    "Damaged pests": 1583,
    "Tungro viral": 1436,
    "Brown spot fungal": 1244,
    "Downy mildew fungal": 1159,
    "Leaf blight bacterial": 1081,
    "Leaf Streak bacterial": 1034
}


augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
])


os.makedirs(target_dir, exist_ok=True)
for class_name in target_counts:
    os.makedirs(os.path.join(target_dir, class_name), exist_ok=True)


for class_name, target in target_counts.items():
    src_folder = os.path.join(source_dir, class_name)
    tgt_folder = os.path.join(target_dir, class_name)

    images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    
    for img_file in images:
        shutil.copy(os.path.join(src_folder, img_file), os.path.join(tgt_folder, img_file))

    
    current_count = len(images)
    needed = target - current_count
    print(f"{class_name}: {current_count} → {target} (augmenting {needed})")

    if needed > 0:
        for i in range(needed):
            img_file = random.choice(images)
            try:
                img_path = os.path.join(src_folder, img_file)
                image = Image.open(img_path).convert('RGB')
                aug_img = augment(image)
                aug_img.save(os.path.join(tgt_folder, f"aug_{i}_{img_file}"))
            except Exception as e:
                print(f"Error augmenting {img_file}: {e}")


print("\n📊 Final image counts after augmentation:")
for class_name in target_counts:
    folder = os.path.join(target_dir, class_name)
    total = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"{class_name}: {total}")
