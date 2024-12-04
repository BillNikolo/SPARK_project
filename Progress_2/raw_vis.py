import pandas as pd
import ast
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Paths to the image folder and CSV file
train_path = r"C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/train"
train_bbox_csv_path = r"C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/labels/train.csv"

# Load bounding boxes from CSV
train_bbox_df = pd.read_csv(train_bbox_csv_path)

# Function to draw the bounding box for a specific image
def draw_bbox_for_image(image_name):
    # Get the image name without extension
    image_name_no_ext = os.path.splitext(image_name)[0]
    
    # Get bounding box for the current image by matching the name without extension
    bbox_data = train_bbox_df[train_bbox_df['filename'] == image_name_no_ext + ".png"]
    if not bbox_data.empty:
        bbox_str = bbox_data['bbox'].values[0]
        bbox_list = ast.literal_eval(bbox_str)
        x_min, y_min, x_max, y_max = bbox_list
        
        # Open the image
        image_path = os.path.join(train_path, image_name)
        with Image.open(image_path) as img:
            # Draw the bounding box
            draw = ImageDraw.Draw(img)
            draw.rectangle([y_min, x_min, y_max, x_max], outline="red", width=2)
            
            # Display the image
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Bounding Box for {image_name}')
            plt.show()
    else:
        print(f"No bounding box found for {image_name}")


# Example usage
if __name__ == "__main__":
    image_name = "img056669.jpg"
    draw_bbox_for_image(image_name)