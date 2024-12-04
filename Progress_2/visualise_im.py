import numpy as np
import os
import matplotlib.pyplot as plt

# Path to the folder containing the numpy batches
batch_path = r"C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/numpy_batches"

# Function to visualize an image and its bounding box from a saved batch by image name
def visualize_image_from_name(image_name):
    # Iterate over all batch files to find the image
    for batch_file in os.listdir(batch_path):
        if batch_file.endswith(".npy"):
            batch_file_path = os.path.join(batch_path, batch_file)
            batch_array = np.load(batch_file_path, allow_pickle=True)
            
            # Search for the image in the current batch
            for index, img_data in enumerate(batch_array):
                if img_data['name'] == image_name:
                    img_array = img_data['image'].astype(np.float32)
                    bbox = img_data['bbox']
                    label = img_data["label"]
                    
                    # Display the image using matplotlib
                    plt.imshow(img_array, vmin=0, vmax=1)
                    plt.axis('off')
                    plt.title(f'Image: {image_name}, label {label}, (Index {index}) from {batch_file}')
                    
                    # Plot bounding box if it exists
                    if bbox is not None:
                        x_min, y_min, x_max, y_max = bbox
                        plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='red', facecolor='none', lw=2))
                    
                    plt.show()
                    return
    
    # If the image is not found in any batch
    print(f"Image {image_name} not found in any batch.")

# Example usage
if __name__ == "__main__":
    # Specify the image name to visualize
    image_name = "img023380.jpg"
    visualize_image_from_name(image_name)

