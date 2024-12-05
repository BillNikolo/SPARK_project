import torch
import torchvision.transforms as T
import numpy as np
import os
import pandas as pd
import ast
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

# Custom dataset to load images and bounding boxes from .npy files
class CustomDataset(Dataset):
    def __init__(self, batch_files, transform=None):
        self.data = []
        for batch_file in batch_files:
            batch_data = np.load(batch_file, allow_pickle=True)
            self.data.extend(batch_data)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        image = img_data['image'].astype(np.float32)  # Convert image array to float32
        bbox = img_data['bbox']
        label = img_data['label']

        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)

        # Create target dictionary with bounding box and label
        target = {}
        if bbox is not None:
            target['boxes'] = torch.tensor([bbox], dtype=torch.float32)
        else:
            target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
        target['labels'] = torch.tensor([label], dtype=torch.int64)

        return image, target

# Load the pretrained ResNet50 model for object detection
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# Replace the classifier head to match our use case (num_classes object classes + background)
num_classes = 12  # Update this based on the number of unique classes in your dataset (11 classes + background)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Paths to the dataset
train_path = r"C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/numpy_batches"
val_path = r"C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/numpy_batches"

# Load training and validation batches
# Specify the train batches to use (provide specific indices or ranges)
train_batch_indices = [0]  # Example: select specific batch indices
train_batches = [os.path.join(train_path, f) for i, f in enumerate(os.listdir(train_path)) if f.endswith('.npy') and i in train_batch_indices]
# Specify the validation batches to use (provide specific indices or ranges)
val_batch_indices = [1]  # Example: select specific batch indices
val_batches = [os.path.join(val_path, f) for i, f in enumerate(os.listdir(val_path)) if f.endswith('.npy') and i in val_batch_indices]

# Define transformations for the input data
transform = T.Compose([
    T.ToTensor(),
])

# Create datasets and data loaders
train_dataset = CustomDataset(train_batches, transform=transform)
val_dataset = CustomDataset(val_batches, transform=transform)


# Set batch size (you can change it based on system capability)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Set device to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Set up the optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training and validation loop
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch+1}/{num_epochs}")
    # Training phase
    for batch_idx, (images, targets) in enumerate(train_loader):
        print(f"Training batch {batch_idx+1}/{len(train_loader)} in epoch {epoch+1}")
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        print(f"Batch {batch_idx+1} Loss: {losses.item():.4f}")

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # Step the learning rate scheduler
    lr_scheduler.step()

    # Validation phase
    model.eval()
    iou_scores = []
    num_samples = 0
    with torch.no_grad():
        print(f"Starting validation for epoch {epoch+1}")
        for batch_idx, (images, targets) in enumerate(val_loader):
            print(f"Validating batch {batch_idx+1}/{len(val_loader)} in epoch {epoch+1}")
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get predictions
            outputs = model(images)

            # Calculate IoU for each image
            for target, output in zip(targets, outputs):
                if len(target['boxes']) > 0 and len(output['boxes']) > 0:
                    # Compute IoU between predicted and ground truth boxes
                    iou_matrix = box_iou(target['boxes'], output['boxes'])
                    iou = iou_matrix.diag().mean().item()  # Average IoU for matched boxes
                    iou_scores.append(iou)
                num_samples += 1

    # Calculate and print average IoU
    avg_iou = np.sum(iou_scores) / num_samples if num_samples > 0 else 0
    print(f"Epoch {epoch+1}/{num_epochs}, Average IoU: {avg_iou:.4f}")

print("Training complete.")
