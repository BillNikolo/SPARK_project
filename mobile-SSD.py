import cv2
import pandas as pd
import os
import numpy as np

# Load the CSV file
train_csv_path = 'path_to_train_csv/train.csv'
train_data = pd.read_csv(train_csv_path)

image_directory = 'path_to_images/'

# Load the pre-trained MobileNet-SSD model and the config file
model_path = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
config_path = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Class labels for the pre-trained model
class_labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Function to preprocess and predict bounding boxes using MobileNet-SSD
def predict_bounding_box(image):
    # Preprocess the image
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    (h, w) = image.shape[:2]

    # Loop over detections and draw bounding boxes
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:  # Minimum confidence threshold
            idx = int(detections[0, 0, i, 1])
            label = class_labels[idx]

            # Get the bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and label on the image
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(image, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Loop through the CSV and process each image
for index, row in train_data.iterrows():
    image_name = row['Image name']
    image_path = os.path.join(image_directory, image_name)
    
    # Load the image
    image = cv2.imread(image_path)
    
    if image is not None:
        # Predict bounding boxes and classify objects
        output_image = predict_bounding_box(image)
        
        # Display the output image (optional)
        # cv2.imshow("Output", output_image)
        # cv2.waitKey(0)
        
        # Save the output image with bounding boxes
        output_image_path = os.path.join('output_images', image_name)
        cv2.imwrite(output_image_path, output_image)
    else:
        print(f"Image {image_name} not found.")

cv2.destroyAllWindows()
