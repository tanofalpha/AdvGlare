# We have done the dataset organization and transformation part on Kaggle
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import random

# Organising The Dataset
# Seed is 42
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Dataset paths
# Dataset link- https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
BASE_PATH = "/kaggle/input/gtsrb-german-traffic-sign"
TRAIN_CSV = os.path.join(BASE_PATH, "Train.csv")
TEST_CSV = os.path.join(BASE_PATH, "Test.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "Train")
TEST_DIR = os.path.join(BASE_PATH, "Test")

# We took the top 16 classes (with respect to the frequency of images of each class)
train_df = pd.read_csv(TRAIN_CSV)
class_counts = train_df['ClassId'].value_counts()
top_classes = class_counts[:16].index.tolist()

filtered_df = train_df[train_df['ClassId'].isin(top_classes)].reset_index(drop=True)
top16_classes = [2, 1, 13, 12, 38, 10, 4, 5, 25, 9, 7, 3, 8, 11, 35, 18]

test_csv_path = '/kaggle/input/gtsrb-german-traffic-sign/Test.csv'
test_dir = '/kaggle/input/gtsrb-german-traffic-sign/Test'
test_df = pd.read_csv(test_csv_path)
test_df = test_df[test_df['ClassId'].isin(top16_classes)]

print(f"Filtered Test Set Size: {len(test_df)}")

# Train Dataset (only preprocessing the top 16 classes)
root_dir = '/kaggle/input/gtsrb-german-traffic-sign/Train'
output_dir = '/kaggle/working/gtsrb_resized'  # Output directory for the new resized interpolated dataset

# Creating output directories for each class
for class_id in filtered_df["ClassId"].unique():
    os.makedirs(os.path.join(output_dir, str(class_id)), exist_ok=True)

# Preprocessing and saving the images
for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
    img_path = os.path.join(root_dir, str(row['ClassId']), row['Path'].split("/")[-1])
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (800, 800), interpolation=cv2.INTER_NEAREST)  # Resizing the dataset
    image = cv2.GaussianBlur(image, (31, 31), 0)  # Gaussian blur

    # Saving the preprocessed image
    save_path = os.path.join(output_dir, str(row['ClassId']), row['Path'].split("/")[-1])
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Saving as BGR for OpenCV compatibility

# Test Dataset (repeating the same steps as the train data)
test_img_dir = '/kaggle/input/gtsrb-german-traffic-sign/Test'
output_dir = '/kaggle/working/gtsrb_resized_test'

# Creating output class folders
for class_id in test_df["ClassId"].unique():
    os.makedirs(os.path.join(output_dir, str(class_id)), exist_ok=True)

# Processing and saving
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    filename = row["Path"].split("/")[-1]
    label = row["ClassId"]
    img_path = os.path.join(test_img_dir, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Could not read image: {img_path}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)
    img = cv2.GaussianBlur(img, (31, 31), 0)

    save_path = os.path.join(output_dir, str(label), filename)
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
