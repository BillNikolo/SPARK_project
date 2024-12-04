import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import ast
import matplotlib.pyplot as plt

# Paths to the image folders
train_path = r"C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/train"
val_path = r"C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/val"
train_bbox_csv_path = r"C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/labels/train.csv"
val_bbox_csv_path = r"C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/labels/val.csv"

# Define class dictionary for mapping class names to integers
class_dict = {
    'smart_1': 1,
    'cheops': 2,
    'lisa_pathfinder': 3,
    'debris': 4,
    'proba_3_ocs': 5,
    'proba_3_csc': 6,
    'soho': 7,
    'earth_observation_sat_1': 8,
    'proba_2': 9,
    'xmm_newton': 10,
    'double_star': 11
}


# Output folder for saving NumPy arrays
output_path = r"C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/numpy_batches"
os.makedirs(output_path, exist_ok=True)

# Load bounding boxes from CSV
train_bbox_df = pd.read_csv(train_bbox_csv_path)
val_bbox_df = pd.read_csv(val_bbox_csv_path)

# Function to process images in batches of 2000

def process_images_in_batches(folder_path, prefix, bbox_df, batch_size=2000):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(image_files)
    
    batch = []
    batch_count = 0

    for idx, image_file in enumerate(image_files):
        image_name_no_ext = os.path.splitext(image_file)[0]  # Get the image name without extension
        image_path = os.path.join(folder_path, image_file)
        
        # Get bounding box and class for the current image by matching the name without extension
        bbox_data = bbox_df[bbox_df['filename'] == image_name_no_ext + ".png"]
        bbox_resized = None  # Default value if no bounding box is found
        label = None
        
        if not bbox_data.empty:
            bbox_str = bbox_data['bbox'].values[0]
            bbox_list = ast.literal_eval(bbox_str)
            y_min, x_min, y_max, x_max = bbox_list  # Correcting the order to [x_min, y_min, x_max, y_max]

            # Open the image
            try:
                with Image.open(image_path) as img:
                    original_width, original_height = img.size

                    # Calculate scaling factors while maintaining aspect ratio
                    img_resized = ImageOps.fit(img, (256, 256), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
                    scale_x = 256 / original_width
                    scale_y = 256 / original_height

                    # Resize the bounding box coordinates
                    x_min = int(x_min * scale_x)
                    y_min = int(y_min * scale_y)
                    x_max = int(x_max * scale_x)
                    y_max = int(y_max * scale_y)
                    bbox_resized = [x_min, y_min, x_max, y_max]

                    # Get the class label for the image
                    class_name = bbox_data['class'].values[0]
                    label = class_dict.get(class_name, 1)  # Default to 1 if class not found
            except UnboundLocalError:
                print(f"Failed to open image: {image_path}")
                continue
        
        # Convert the image to a NumPy array and normalize pixel values
        if 'img_resized' in locals():
            img_array = np.array(img_resized, dtype=np.float16) / 255.0  # Normalize pixel values to be between 0 and 1
            batch.append({'image': img_array, 'name': image_file, 'bbox': bbox_resized, 'label': label})
            del img_resized  # Clean up the variable to avoid issues in the next iteration
        
        # If the batch is full or it's the last image, save the batch
        if (idx + 1) % batch_size == 0 or (idx + 1) == num_images:
            batch_array = np.array(batch, dtype=object)
            batch_filename = f"{prefix}_batch_{batch_count}.npy"
            batch_filepath = os.path.join(output_path, batch_filename)
            
            # Save the batch as a .npy file
            np.save(batch_filepath, batch_array)
            print(f"Saved {batch_filename} with shape {batch_array.shape}")
            
            # Clear the batch and increment the batch count
            batch = []
            batch_count += 1

# Process training images
process_images_in_batches(train_path, prefix="train", bbox_df=train_bbox_df)

# Process validation images
process_images_in_batches(val_path, prefix="val", bbox_df=val_bbox_df)