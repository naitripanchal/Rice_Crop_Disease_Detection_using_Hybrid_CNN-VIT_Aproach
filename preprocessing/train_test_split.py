# TO SPLIT BALANCED DATASET INTO TRAIN, TEST AND VALIDATION SET

import os
import shutil
import random
from tqdm import tqdm

source_dir = "../dataset/train_images_balanced"
output_dir = "../dataset/split_dataset"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

random.seed(42)

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

for split in ['train', 'val', 'test']:
    for class_name in os.listdir(source_dir):
        split_path = os.path.join(output_dir, split, class_name)
        os.makedirs(split_path, exist_ok=True)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]


    for img_file in tqdm(train_imgs, desc=f"Train - {class_name}"):
        shutil.copy(os.path.join(class_path, img_file), os.path.join(output_dir, 'train', class_name, img_file))
    for img_file in tqdm(val_imgs, desc=f"Val - {class_name}"):
        shutil.copy(os.path.join(class_path, img_file), os.path.join(output_dir, 'val', class_name, img_file))
    for img_file in tqdm(test_imgs, desc=f"Test - {class_name}"):
        shutil.copy(os.path.join(class_path, img_file), os.path.join(output_dir, 'test', class_name, img_file))

print("\n✅ Dataset successfully split into train, val, and test folders.")
