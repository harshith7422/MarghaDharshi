import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from sklearn.model_selection import train_test_split
import shutil

# Define dataset path
DATASET_PATH = "/Users/harshithyvs/Desktop/VIT/MarghaDharshi/Soil_Module/soil_images"
PROCESSED_DATASET_PATH = "/Users/harshithyvs/Desktop/VIT/MarghaDharshi/Soil_Module/pdata"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Remove existing processed dataset and create fresh folders
if os.path.exists(PROCESSED_DATASET_PATH):
    shutil.rmtree(PROCESSED_DATASET_PATH)
os.makedirs(PROCESSED_DATASET_PATH)

# Create 'train' and 'val' subfolders
for split in ['train', 'val']:
    os.makedirs(os.path.join(PROCESSED_DATASET_PATH, split), exist_ok=True)

# Data Augmentation
data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80-20 split
)

# Process and split images
for category in os.listdir(DATASET_PATH):
    category_path = os.path.join(DATASET_PATH, category)
    if not os.path.isdir(category_path):
        continue
    
    images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith((".jpg", ".png"))]
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    
    # Ensure category-specific directories exist before copying
    train_category_path = os.path.join(PROCESSED_DATASET_PATH, 'train', category)
    val_category_path = os.path.join(PROCESSED_DATASET_PATH, 'val', category)
    os.makedirs(train_category_path, exist_ok=True)
    os.makedirs(val_category_path, exist_ok=True)

    for img_path in train_imgs:
        shutil.copy(img_path, train_category_path)
    for img_path in val_imgs:
        shutil.copy(img_path, val_category_path)

print("✅ Data preprocessing completed!")
