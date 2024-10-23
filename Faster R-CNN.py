import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision
from engine import train_one_epoch, evaluate
import utils
import torchvision.models.detection as models
from torchvision.ops import box_iou

# Custom Dataset for loading images and bounding boxes
class CustomDataset(Dataset):
    def __init__(self, csv_file, image_dir, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, idx):
        # Load image and bounding box
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        
        # Get bounding box and convert to tensor
        box = eval(self.data.iloc[idx, 3])
        bbox = torch.tensor([box], dtype=torch.float32)
        
        # Get class label and convert to tensor
        label = self.data.iloc[idx, 2]
        label_idx = label_map[label]  # Map label to integer index
        labels = torch.tensor([label_idx], dtype=torch.int64)
        
        target = {}
        target['boxes'] = bbox
        target['labels'] = labels
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, target

    def __len__(self):
        return len(self.data)

# Transformations for the dataset
transform = T.Compose([
    T.ToTensor()
])

# Label mapping: Assign an index for each class label
label_map = {
    "Proba3": 1,
    "ObservationSat1": 2,
    "Cheops": 3,
    "Proba3ocs": 4,
    "Smart1": 5,
    "XMM Newton": 6,
    "LisaPathfinder": 7
}

# Prepare Dataset
train_csv = 'path_to_train_csv/train.csv'
train_image_dir = 'path_to_images/'

dataset = CustomDataset(csv_file=train_csv, image_dir=train_image_dir, transforms=transform)

# DataLoader
train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

# Load Faster R-CNN pre-trained on COCO
model = models.fasterrcnn_resnet50_fpn(pretrained=True)

# Get the number of input features for the classifier
num_classes = len(label_map) + 1  # Add 1 for the background class

# Replace the classifier head to match the number of classes in your dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Number of epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    lr_scheduler.step()

    # Evaluate the model on the validation set
    evaluate(model, train_loader, device=device)

# Save the trained model
torch.save(model.state_dict(), "fasterrcnn_custom.pth")

# Load test dataset similarly to the training set
test_csv = 'path_to_test_csv/test.csv'
test_image_dir = 'path_to_test_images/'

test_dataset = CustomDataset(csv_file=test_csv, image_dir=test_image_dir, transforms=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

# Load the trained model
model.load_state_dict(torch.load("fasterrcnn_custom.pth"))
model.eval()

# Run predictions on the test dataset
for images, targets in test_loader:
    images = [image.to(device) for image in images]
    outputs = model(images)  # Get predictions

    # Display the predicted bounding boxes and labels for each image
    for i, output in enumerate(outputs):
        print(f"Image {i}:")
        print("Boxes:", output['boxes'])
        print("Labels:", output['labels'])

# For a single prediction
pred_boxes = outputs[0]['boxes']
true_boxes = targets[0]['boxes']

iou = box_iou(pred_boxes, true_boxes)
print(f"Intersection over Union (IoU): {iou}")
