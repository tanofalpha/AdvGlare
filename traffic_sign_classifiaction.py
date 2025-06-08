import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BATCH_SIZE = 64
EPOCHS = 5
TARGET_CLASSES = 16

# Dataset paths (Kaggle links as comments)
# https://www.kaggle.com/datasets/akshat1012/gtsrb-resized-test-data
# https://www.kaggle.com/datasets/akshat1012/gtsrb-resized-train-data
# https://www.kaggle.com/datasets/akshat1012/gtsrb-csv-data

BASE_PATH = "/kaggle/input/gtsrb-csv-data"
TRAIN_CSV = os.path.join(BASE_PATH, "Train.csv")
TRAIN_DIR = "/kaggle/input/gtsrb-resizedtrain-data"

df = pd.read_csv(TRAIN_CSV)
class_counts = df['ClassId'].value_counts()
top_classes = sorted(class_counts[:TARGET_CLASSES].index.tolist())

class_id_to_index = {orig: idx for idx, orig in enumerate(top_classes)}
index_to_class_id = {v: k for k, v in class_id_to_index.items()}

print("Label Mapping:")
for k, v in class_id_to_index.items():
    print(f"Original label {k} index {v}")

filtered_df = df[df['ClassId'].isin(top_classes)].reset_index(drop=True)
filtered_df["ClassId"] = filtered_df["ClassId"].map(class_id_to_index)

# Dataset class
class GTSRBDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.data = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = row['ClassId']
        original_class_folder = index_to_class_id[label]
        img_path = os.path.join(self.root_dir, str(original_class_folder), row['Path'].split("/")[-1])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label

# Resize lookup for different models
resize_lookup = {
    "resnet50": (224, 224),
    "densenet121": (224, 224),
    "mobilenet_v2": (224, 224),
}

# Model definition function
def get_model(name, num_classes):
    if name == "resnet50":
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        model = models.densenet121(weights='IMAGENET1K_V1')
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported model: " + name)
    return model.to(device)

# Training function
def train(model, loader, val_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        acc = correct / len(loader.dataset)
        print(f"Epoch {epoch + 1} - Loss: {total_loss:.4f}, Accuracy: {acc * 100:.2f}%")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, val_data = train_test_split(
    filtered_df, test_size=0.2, stratify=filtered_df["ClassId"], random_state=SEED
)

# Train and save models
model_names = ["resnet50", "densenet121", "mobilenet_v2"]
for model_name in model_names:
    print(f"\n++++ Training {model_name.upper()} ++++")
    resize_dims = resize_lookup[model_name]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize_dims),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = GTSRBDataset(train_data, TRAIN_DIR, transform)
    val_dataset = GTSRBDataset(val_data, TRAIN_DIR, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model(model_name, num_classes=TARGET_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, val_loader, optimizer, criterion, epochs=EPOCHS)

    model_path = f"/kaggle/working/{model_name}_gtsrb_top{TARGET_CLASSES}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_name} to {model_path}")

# Evaluation
TEST_DIR = "/kaggle/input/gtsrb-resizedtest-data"
TRAIN_CSV = "/kaggle/input/gtsrb-csv-data/Train.csv"
MODEL_DIR = "/kaggle/working/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(TRAIN_CSV)
class_counts = df['ClassId'].value_counts()
top_classes = sorted(class_counts[:TARGET_CLASSES].index.tolist())
class_id_to_index = {orig: idx for idx, orig in enumerate(top_classes)}
index_to_class_id = {v: k for k, v in class_id_to_index.items()}

resize_dims = (224, 224)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(resize_dims),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class GTSRBTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for orig_class_id in top_classes:
            class_dir = os.path.join(root_dir, str(orig_class_id))
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                mapped_label = class_id_to_index[orig_class_id]
                self.samples.append((img_path, mapped_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label

def get_model(name, num_classes):
    if name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model.to(DEVICE)

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Evaluate saved models
test_dataset = GTSRBTestDataset(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for model_name in model_names:
    print(f"\n==== Evaluating {model_name.upper()} ====")
    model = get_model(model_name, num_classes=TARGET_CLASSES)
    model_path = os.path.join(MODEL_DIR, f"{model_name}_gtsrb_top{TARGET_CLASSES}.pth")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        continue
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    evaluate(model, test_loader)
