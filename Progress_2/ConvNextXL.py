import torch
import torchvision.transforms as T
import numpy as np
import os
import pandas as pd
import ast
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import convnext_large
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch import nn

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

# Load the pretrained ConvNeXt Large model as a backbone
convnext_backbone = convnext_large(pretrained=True)

# Wrap the ConvNeXt backbone to extract features and add a convolutional layer to match RPN input requirements
class BackboneWithFPN(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = nn.Sequential(*list(backbone.features.children())[:-2])  # Use all layers except the final classifier
        self.conv = nn.Conv2d(768, 1024, kernel_size=1)  # Add convolutional layer to adjust channel count to 1024
        self.out_channels = 1024

    def forward(self, x):
        features = self.backbone(x)
        features = self.conv(features)  # Adjust output channels to 1024
        return {'0': features}

backbone = BackboneWithFPN(convnext_backbone)

# Define the region proposal network's anchor generator
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),) * len(((32, 64, 128, 256, 512),))
)

# Define the RoI aligner
import torchvision

roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'],
    output_size=7,
    sampling_ratio=2
)

# Create the Faster R-CNN model using the ConvNeXt backbone
model = FasterRCNN(
    backbone,
    num_classes=12,  # Update this based on the number of unique classes in your dataset (11 classes + background)
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
)

# Paths to the dataset
train_path = r"C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/numpy_batches"
val_path = r"C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/numpy_batches"

# Load training and validation batches
# Specify the train batches to use (provide specific indices or ranges)
train_batch_indices = [0, 2]  # Example: select specific batch indices
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
batch_size = 100

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

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
    y_true = []
    y_pred = []
    with torch.no_grad():
        print(f"Starting validation for epoch {epoch+1}")
        for batch_idx, (images, targets) in enumerate(val_loader):
            print(f"Validating batch {batch_idx+1}/{len(val_loader)} in epoch {epoch+1}")
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get predictions
            outputs = model(images)

            for target, output in zip(targets, outputs):
                true_labels = target['labels'].cpu().numpy()
                y_true.extend(true_labels)

                if len(output['labels']) > 0:
                    pred_labels = output['labels'].cpu().numpy()
                    y_pred.extend(pred_labels)
                else:
                    y_pred.extend([0] * len(true_labels))

                if len(target['boxes']) > 0 and len(output['boxes']) > 0:
                    iou_matrix = box_iou(target['boxes'], output['boxes'])
                    iou = iou_matrix.diag().mean().item()
                    iou_scores.append(iou)

    # Metrics calculation
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    au_roc = roc_auc_score(y_true, y_pred, multi_class='ovr') if len(set(y_true)) > 1 else 0

    avg_iou = np.mean(iou_scores) if iou_scores else 0
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AU-ROC: {au_roc:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")

print("Training complete.")