import os
import sys
import random
import time
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import timm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Using device: {device}")

# 1. Transforms (Tinh gọn, giữ nguyên đặc trưng vật liệu)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# 2. Dataset
class TrashDataset(Dataset):
    def __init__(self, root_dir, df=None, csv_file_name=None, transform=None, is_test=False):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.transform = transform
        self.is_test = is_test

        if is_test:
            self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            self.annotations = pd.DataFrame({'file_name': self.image_files})
        elif df is not None:
            self.annotations = df.reset_index(drop=True)
        else:
            self.annotations = pd.read_csv(os.path.join(root_dir, csv_file_name))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.img_dir, row['file_name'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, row['file_name']

        label = int(row['category_id']) - 1
        return image, label

# 3. Data Split & Loaders
train_dir = "data/TACVU2/train"
df_full = pd.read_csv(os.path.join(train_dir, "train.csv"))
train_df, val_df = train_test_split(
    df_full, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_full['category_id']
)

BATCH_SIZE = 32
train_dataset = TrashDataset(root_dir=train_dir, df=train_df, transform=train_transforms)
val_dataset = TrashDataset(root_dir=train_dir, df=val_df, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 4. Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class_counts = train_df['category_id'].value_counts().sort_index().values
class_counts = torch.tensor(class_counts, dtype=torch.float32)
weights = 1.0 / (class_counts ** 0.5)
alpha = (weights / weights.sum()).to(device)

# 5. Model: Mở bung sức (drop_rate=0.0, drop_path_rate=0.0)
print("Khởi tạo EfficientNet-B2 UNLEASHED (drop_rate=0.0, drop_path_rate=0.0)...")
model = timm.create_model(
    'efficientnet_b2', 
    pretrained=True, 
    num_classes=6,
    drop_rate=0.0,
    drop_path_rate=0.0
)
model.to(device)

criterion = FocalLoss(alpha=alpha, gamma=1.5)
# Learning rate 3e-4, weight_decay giảm xuống 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

NUM_EPOCH = 20
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCH, eta_min=1e-6)

best_model_path = "data/TACVU2/best_model_eff_b2.pth"
best_macro_f1 = 0.0

print(f"\n=== BẮT ĐẦU HUẤN LUYỆN {NUM_EPOCH} EPOCHS (MỞ HẾT SỨC) ===")

for epoch in range(NUM_EPOCH):
    start_time = time.time()
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = 100.0 * correct / total

    # Validation
    model.eval()
    val_running_loss = 0.0
    all_predicts, all_targets = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)

            all_predicts.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    val_loss = val_running_loss / len(val_dataset)
    val_acc = 100.0 * (np.array(all_predicts) == np.array(all_targets)).mean()
    val_macro_f1 = f1_score(all_targets, all_predicts, average="macro")

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    elapsed = time.time() - start_time

    best_msg = ""
    if val_macro_f1 > best_macro_f1:
        best_macro_f1 = val_macro_f1
        torch.save(model.state_dict(), best_model_path)
        best_msg = f"  ===> [LƯU KỶ LỤC MỚI: {val_macro_f1*100:.2f}%]"

    print(f"Epoch [{epoch + 1:02d}/{NUM_EPOCH}] ({elapsed:.1f}s) | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_macro_f1*100:.2f}%{best_msg}")
    sys.stdout.flush()

print(f"\n Huấn luyện hoàn thành! Kỷ lục Macro F1: {best_macro_f1*100:.2f}%")

# 6. Post-processing: Tối ưu hóa ngưỡng hậu xử lý (Threshold Tuning) & Inference Private Test
print("\n=== TIẾN HÀNH THRESHOLD TUNING & INFERENCE TRÊN PRIVATE TEST ===")
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

val_probs, val_targets = [], []
with torch.no_grad():
    for images, targets in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        val_probs.append(probs)
        val_targets.append(targets.numpy())

val_probs = np.vstack(val_probs)
val_targets = np.concatenate(val_targets)

# Coordinate search tìm weights
best_w = np.ones(6)
best_tuned_f1 = f1_score(val_targets, np.argmax(val_probs, axis=1), average='macro')
for it in range(5):
    improved = False
    for c in range(6):
        for cand in np.linspace(0.5, 2.5, 41):
            tw = best_w.copy()
            tw[c] = cand
            score = f1_score(val_targets, np.argmax(val_probs * tw, axis=1), average='macro')
            if score > best_tuned_f1 + 1e-4:
                best_tuned_f1 = score
                best_w = tw.copy()
                improved = True
    if not improved:
        break

print(f"Macro F1 trước Tuning: {f1_score(val_targets, np.argmax(val_probs, axis=1), average='macro')*100:.2f}%")
print(f"Macro F1 sau Tuning:   {best_tuned_f1*100:.2f}%")
print(f"Bộ Multipliers w: {[round(x, 4) for x in best_w.tolist()]}")

# Inference Private Test với TTA + Threshold Tuning
private_dir = "data/TACVU2/private_test"
test_dataset = TrashDataset(root_dir=private_dir, transform=val_transforms, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

file_names, predictions = [], []

with torch.no_grad():
    for images, fnames in test_loader:
        images = images.to(device)
        # TTA: ảnh gốc + lật ngang
        outputs_orig = model(images)
        outputs_flip = model(torch.flip(images, dims=[3]))
        avg_probs = (torch.softmax(outputs_orig, dim=1).cpu().numpy() + torch.softmax(outputs_flip, dim=1).cpu().numpy()) / 2.0
        
        preds = np.argmax(avg_probs * best_w, axis=1)
        file_names.extend(fnames)
        predictions.extend(preds + 1)

df_sub = pd.DataFrame({"file_name": file_names, "category_id": predictions})
sub_path = "data/TACVU2/submission.csv"
df_sub.to_csv(sub_path, index=False)
print(f"\nĐã xuất file {sub_path} với {len(df_sub)} dự đoán!")
print("Phân phối nhãn dự đoán:")
print(df_sub['category_id'].value_counts().sort_index())

# Đóng gói submission.zip
with zipfile.ZipFile("data/TACVU2/submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write("data/TACVU2/submission.csv", arcname="submission.csv")
    zf.write("data/TACVU2/generate_result.ipynb", arcname="generate_result.ipynb")

import shutil
shutil.copyfile("data/TACVU2/submission.zip", "submission.zip")
print(" Đã cập nhật xong submission.zip sẵn sàng nộp!")
