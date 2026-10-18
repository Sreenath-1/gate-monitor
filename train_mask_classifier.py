import os
import glob
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import onnxruntime as ort

# ==============================================================================
# CONFIG
# ==============================================================================
IMG_SIZE = 112 # Same as ArcFace crops to keep it standard
BATCH_SIZE = 32
EPOCHS = 6 # Fine-tune powerful backbone
MODEL_EXPORT_PATH = "weights/mask_classifier.onnx"

# HuggingFace dataset labels:
# 0: WithMask, 1: WithoutMask
# We map to: 1 (masked) -> 1, 1 (without) -> 0

# ==============================================================================
# PREPROCESSING
# ==============================================================================
# We must use EXACTLY the preprocessing the inference pipeline will use.
# For ONNX model, we typically do: Resize -> RGB -> (x/255.0 - mean)/std
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# ==============================================================================
# DATASET LOADING
# ==============================================================================
print("Loading HuggingFace FaceMask dataset...")
dataset = load_dataset("sumitpardhiya/Face-Mask-Detection")

class HFMaskDataset(Dataset):
    def __init__(self, hf_ds, transform=None):
        self.hf_ds = hf_ds
        self.transform = transform
        
    def __len__(self):
        return len(self.hf_ds)
        
    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        image = item['image'].convert('RGB')
        label = item['label']
        # Map 0 (WithMask) -> 1 (Masked), 1 (WithoutMask) -> 0 (Unmasked)
        binary_label = 1 if label == 0 else 0
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(binary_label, dtype=torch.float32)

# Use both train and test splits for training data to get 400 images
from torch.utils.data import ConcatDataset
train_ds = ConcatDataset([HFMaskDataset(dataset['train'], train_transform), HFMaskDataset(dataset['test'], train_transform)])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ==============================================================================
# DOMAIN VALIDATION SET (from known_faces/)
# ==============================================================================
def get_domain_validation_data():
    x_val = []
    y_val = []
    bucket_info = []
    for user_dir in glob.glob("known_faces/*"):
        if not os.path.isdir(user_dir): continue
        for img_path in glob.glob(os.path.join(user_dir, "*.jpg")):
            img = cv2.imread(img_path)
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Determine label from filename bucket info (or just folder structure)
            fname = os.path.basename(img_path).lower()
            
            # Read bucket metadata if available to be exact
            bucket = "unknown"
            meta_path = os.path.join(user_dir, "bucket_metadata.json")
            if os.path.exists(meta_path):
                import json
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    for frame in meta.get("frames", []):
                        if frame["filename"] == os.path.basename(img_path):
                            bucket = frame["bucket"]
                            break
                            
            if bucket == "unknown":
                if "mask" in user_dir.lower():
                    bucket = "masked_frontal" # guess
                else:
                    bucket = "frontal_unmasked"

            label = 1 if "masked" in bucket and "unmasked" not in bucket else 0
            
            x_val.append(img_rgb)
            y_val.append(label)
            bucket_info.append(bucket)
            
    return x_val, y_val, bucket_info

print("Loading domain validation data...")
x_domain, y_domain, buckets_domain = get_domain_validation_data()

# ==============================================================================
# MODEL DEFINITION
# ==============================================================================
print("Initializing MobileNetV2...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
model.classifier[1] = nn.Linear(model.last_channel, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# ==============================================================================
# TRAINING LOOP
# ==============================================================================
print(f"Training on {device}...")
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader.dataset):.4f}")

# ==============================================================================
# ONNX EXPORT
# ==============================================================================
print("Exporting to ONNX...")
model.eval()
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

if not os.path.exists("weights"):
    os.makedirs("weights")

torch.onnx.export(
    model, 
    dummy_input, 
    MODEL_EXPORT_PATH,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f"Exported to {MODEL_EXPORT_PATH}")

# ==============================================================================
# DOMAIN VALIDATION (ONNX INFERENCE)
# ==============================================================================
print("Running domain validation via ONNXRuntime...")
ort_session = ort.InferenceSession(MODEL_EXPORT_PATH, providers=['CPUExecutionProvider'])

def preprocess_for_onnx(img_rgb):
    # Same as val_transform but manual
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    img = np.transpose(img, (2, 0, 1)) # HWC to CHW
    img = np.expand_dims(img, axis=0) # add batch
    return img

correct = 0
total = len(x_domain)
bucket_stats = {}
latency_sum = 0

for i in range(total):
    img_rgb = x_domain[i]
    true_label = y_domain[i]
    bucket = buckets_domain[i]
    
    if bucket not in bucket_stats:
        bucket_stats[bucket] = {"total": 0, "correct": 0}
        
    inp = preprocess_for_onnx(img_rgb)
    
    t0 = time.time()
    ort_inputs = {ort_session.get_inputs()[0].name: inp}
    ort_outs = ort_session.run(None, ort_inputs)
    t1 = time.time()
    latency_sum += (t1 - t0)
    
    # Sigmoid for binary prediction
    pred_logit = ort_outs[0][0][0]
    pred_prob = 1.0 / (1.0 + np.exp(-pred_logit))
    pred_label = 1 if pred_prob > 0.5 else 0
    
    bucket_stats[bucket]["total"] += 1
    if pred_label == true_label:
        bucket_stats[bucket]["correct"] += 1
        correct += 1
    else:
        print(f"  [MISS] Bucket {bucket} | True={true_label}, Pred={pred_prob:.3f}")

print("\n--- VALIDATION RESULTS ---")
print(f"Total Domain Images : {total}")
if total > 0:
    print(f"Overall Accuracy    : {correct/total*100:.1f}%")
    print(f"Avg CPU Latency     : {(latency_sum/total)*1000:.2f} ms")
    
    for b, stats in bucket_stats.items():
        acc = stats["correct"] / stats["total"] * 100
        print(f"  [{b}]: {stats['correct']}/{stats['total']} ({acc:.1f}%)")

