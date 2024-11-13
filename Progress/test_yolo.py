from ultralytics import YOLO
import os
import cv2

# Load the saved YOLO model
model = YOLO("saved_yolo_model.pt")

# Path to test images
test_images_path = 'C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/test'

# Function to detect and display results on test images
def detect_and_display(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
        label = model.names[int(result.cls[0])]
        confidence = result.conf[0]

        # Draw bounding boxes
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Detection", image)
    cv2.waitKey(0)

# Run detection on each test image
for filename in os.listdir(test_images_path):
    file_path = os.path.join(test_images_path, filename)
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        detect_and_display(file_path)

cv2.destroyAllWindows()
