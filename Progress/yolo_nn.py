from ultralytics import YOLO
import os
import pandas as pd
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Define paths
train_images_path = 'C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/train'
val_images_path = 'C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/val'
train_csv_path = 'C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/labels/train.csv'
val_csv_path = 'C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/labels/val.csv'

# Load YOLO model
model = YOLO("yolov5s.pt")  # YOLOv5 small model pre-trained on COCO

# Load train and validation labels
train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

# Helper function to detect objects in images
def detect_objects(image_path):
    image = cv2.imread(image_path)
    results = model(image)  # Perform object detection
    return results[0]  # Return detection results for this image

# Evaluate on validation set and collect metrics
y_true, y_pred = [], []
for _, row in val_df.iterrows():
    filename, true_label = row['filename'], row['label']
    image_path = os.path.join(val_images_path, filename)
    if os.path.exists(image_path):
        result = detect_objects(image_path)
        detected_labels = [model.names[int(box.cls[0])] for box in result.boxes]  # Get detected labels
        
        # If there's a detected label matching the true label, count as correct
        if true_label in detected_labels:
            y_pred.append(true_label)
        else:
            y_pred.append(detected_labels[0] if detected_labels else 'unknown')
        y_true.append(true_label)

# Calculate metrics
print("Classification Report:")
print(classification_report(y_true, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Save the YOLO model
model_path = "saved_yolo_model.pt"
model.save(model_path)
print(f"Model saved to {model_path}")

