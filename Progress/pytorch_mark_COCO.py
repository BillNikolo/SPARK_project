import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.ops import box_iou
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from PIL import Image
import pandas as pd
import os
import ast

# Dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_dir, transforms=None):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transforms = transforms
        self.valid_extensions = ['.jpg', '.jpeg', '.png']  # Supported extensions

    def _find_correct_extension(self, filename):
        """
        Check for the correct file extension by testing valid extensions.
        """
        for ext in self.valid_extensions:
            img_path = os.path.join(self.image_dir, os.path.splitext(filename)[0] + ext)
            if os.path.exists(img_path):
                return img_path
        raise FileNotFoundError(f"Image file not found with any valid extension: {filename}")

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        filename = row['filename']

        # Dynamically determine the correct extension
        img_path = self._find_correct_extension(filename)

        # Debugging: Print the resolved path
        print(f"Resolved image path: {img_path}")

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error opening image {img_path}: {e}")
        
        # Parse the bounding box column
        bbox = ast.literal_eval(row['bbox'])  # Convert string to list
        xmin, ymin, xmax, ymax = bbox

        # Prepare bounding boxes and labels
        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        labels = torch.tensor([row['class']], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        # Apply transforms
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.annotations)

# Transforms
class ToTensor:
    def __call__(self, img):
        return F.to_tensor(img)

# Create datasets and dataloaders
train_dataset = CustomDataset(csv_file='C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/labels/train.csv', image_dir='C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/train/', transforms=ToTensor())
val_dataset = CustomDataset(csv_file='C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/labels/val.csv', image_dir='C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/val/', transforms=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Model
def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Initialize model
num_classes = len(train_dataset.annotations['class'].unique()) + 1
model = get_model(num_classes)

# IoU Calculation Example (Torchvision Built-in)
def calculate_iou(pred_boxes, target_boxes):
    return box_iou(pred_boxes, target_boxes)

# Metrics
mean_ap = MeanAveragePrecision()

# Validation loop with built-in metrics
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

model.eval()
with torch.no_grad():
    for images, targets in val_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Get predictions
        outputs = model(images)

        # Format data for metrics
        preds = [
            {
                "boxes": out["boxes"],
                "scores": out["scores"],
                "labels": out["labels"],
            }
            for out in outputs
        ]

        target_metrics = [
            {
                "boxes": t["boxes"],
                "labels": t["labels"],
            }
            for t in targets
        ]

        # Update metrics
        mean_ap.update(preds, target_metrics)

# Compute Mean Average Precision (mAP)
final_map = mean_ap.compute()
print("Mean Average Precision (mAP):", final_map)
