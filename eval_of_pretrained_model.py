import os
from glob import glob
from collections import defaultdict

# Datasets
# "/kaggle/input/gtsrb-resized-test-data"
# "/kaggle/input/gtsrb-test-attacked"
# "/kaggle/input/final-trained-models-resized-gtsrb-dataset"

original_test_dir = "/kaggle/input/gtsrb-resized-test-data"
attacked_test_dir = "/kaggle/input/gtsrb-test-attacked/gtsrb_resized_test_advAttacked+++++"

# Collect all attacked images and their relative paths (example: 35/00085.png)
attacked_image_paths = glob(os.path.join(attacked_test_dir, "*", "*.png"))
attacked_relative_paths = [os.path.relpath(path, attacked_test_dir) for path in attacked_image_paths]

# Now we find the corresponding original test image paths
original_image_paths = [os.path.join(original_test_dir, rel_path) for rel_path in attacked_relative_paths]

# Example
print("Sample attacked image path:", attacked_image_paths[0])
print("Corresponding original image path:", original_image_paths[0])
print(f"Total matched images: {len(original_image_paths)}")

import pandas as pd
TRAIN_CSV = '/kaggle/input/gtsrb-csv-data/Train.csv'
df = pd.read_csv(TRAIN_CSV)

# Get the most frequent classes (you can adjust TARGET_CLASSES to be the number of classes you want)
TARGET_CLASSES = 16
class_counts = df['ClassId'].value_counts()

# Get the top classes and sort them for consistency (we did this previously too while training the model)
top_classes = sorted(class_counts[:TARGET_CLASSES].index.tolist())

# Create label remapping (original labels to new indices)
class_id_to_index = {orig: idx for idx, orig in enumerate(top_classes)}
index_to_class_id = {v: k for k, v in class_id_to_index.items()}

# Printing the label mapping
print("Label Mapping:")
for k, v in class_id_to_index.items():
    print(f"Original label {k} -> New index {v}")

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2

# Loading the original and attacked image as well as their labels to use for evaluating/testing
class GTSRBPairDataset(Dataset):
    def __init__(self, original_image_paths, attacked_image_paths, transform=None):
        self.original_image_paths = original_image_paths
        self.attacked_image_paths = attacked_image_paths
        self.transform = transform

    def __len__(self):
        return len(self.original_image_paths)

    def __getitem__(self, idx):
        # Load original and attacked image
        orig_path = self.original_image_paths[idx]
        adv_path = self.attacked_image_paths[idx]

        orig_img = cv2.imread(orig_path)
        adv_img = cv2.imread(adv_path)

        # Convert BGR to RGB
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        adv_img = cv2.cvtColor(adv_img, cv2.COLOR_BGR2RGB)

        # Getting the label from the folder name (it is the label itself)
        label = int(orig_path.split('/')[-2])

        if self.transform:
            orig_img = self.transform(orig_img)
            adv_img = self.transform(adv_img)

        return orig_img, adv_img, label, orig_path.split("/")[-2] + "/" + orig_path.split("/")[-1]

# Defining transforms (matching training config)
resize_dims = (224, 224)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(resize_dims),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialising dataset
dataset = GTSRBPairDataset(original_image_paths, attacked_image_paths, transform=transform)
orig_img, adv_img, label, img_name = dataset[0]

print(f"Label: {label}, Image name: {img_name}")
print(f"Original image shape: {orig_img.shape}, Attacked image shape: {adv_img.shape}")

import torch.nn as nn
from torchvision import models

# Loading the models on the basis of what model we want to test (resnet50, densenet121, mobilenetv2)
def load_model(pth_path):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 16)
    # model = models.mobilenet_v2(weights=None)
    # model.classifier[1] = nn.Linear(model.classifier[1].in_features, 16)
    # model = models.densenet121(weights=None)
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()
    return model

# Path to model .pth file (change it according to the model we are testing)
model_path = "/kaggle/input/final-trained-models-resized-gtsrb-dataset/resnet50_gtsrb_top16.pth"
model = load_model(model_path)

# Using the GPU on Kaggle (GPU T4 x2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

# Creating the DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Saving the values in list to make histograms later
original_preds = []
attacked_preds = []
true_labels = []
image_ids = []
original_confidences = []
attacked_confidences = []

with torch.no_grad():
    for orig_imgs, adv_imgs, labels, img_paths in tqdm(loader):
        orig_imgs = orig_imgs.to(device)
        adv_imgs = adv_imgs.to(device)
        labels = labels.to(device)

        # Predictions and probabilities
        orig_outputs = model(orig_imgs)
        adv_outputs = model(adv_imgs)
        orig_probs = F.softmax(orig_outputs, dim=1)
        adv_probs = F.softmax(adv_outputs, dim=1)

        orig_pred = orig_probs.argmax(dim=1)
        adv_pred = adv_probs.argmax(dim=1)

        original_preds.extend(orig_pred.cpu().tolist())
        attacked_preds.extend(adv_pred.cpu().tolist())
        true_labels.extend(labels.cpu().tolist())
        image_ids.extend(img_paths)
        original_confidences.extend(orig_probs.max(dim=1)[0].cpu().tolist())
        attacked_confidences.extend(adv_probs.max(dim=1)[0].cpu().tolist())

from sklearn.metrics import accuracy_score

true_labels_remapped = [class_id_to_index.get(str(label), -1) for label in true_labels]
original_pred_label_remapped = [class_id_to_index.get(str(pred), -1) for pred in original_preds]
attacked_pred_label_remapped = [class_id_to_index.get(str(pred), -1) for pred in attacked_preds]

# Now calculating the accuracy after remapping
correct_original_preds = sum([
    1 if true_labels_remapped[i] == original_pred_label_remapped[i] else 0
    for i in range(len(true_labels_remapped))
])

correct_attacked_preds = sum([
    1 if true_labels_remapped[i] == attacked_pred_label_remapped[i] else 0
    for i in range(len(true_labels_remapped))
])

# Finding original accuracy, attacked accuracy
original_accuracy = correct_original_preds / len(true_labels_remapped) * 100
attacked_accuracy = correct_attacked_preds / len(true_labels_remapped) * 100

print(f"Original Accuracy: {original_accuracy:.2f}%")
print(f"Attacked Accuracy: {attacked_accuracy:.2f}%")

import matplotlib.pyplot as plt

# Plotting the histograms for original and attacked image confidence scores
plt.figure(figsize=(12, 6))

# Original confidence histogram
plt.subplot(1, 2, 1)
plt.hist(original_confidences, bins=20, color='blue', alpha=0.7)
plt.title('Original Image Confidence Scores')
plt.xlabel('Confidence')
plt.ylabel('Frequency')

# Attacked confidence histogram
plt.subplot(1, 2, 2)
plt.hist(attacked_confidences, bins=20, color='red', alpha=0.7)
plt.title('Attacked Image Confidence Scores')
plt.xlabel('Confidence')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

import numpy as np

# Calculate average confidence scores
avg_confidence_original = np.mean(original_confidences)
avg_confidence_attacked = np.mean(attacked_confidences)

print(f"\nAverage Confidence on Original Images: {avg_confidence_original:.4f}")
print(f"Average Confidence on Attacked Images: {avg_confidence_attacked:.4f}")
