import tensorflow as tf
import pandas as pd
import os

# Load the CSV files
train_labels = pd.read_csv('C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/labels/train.csv')
val_labels = pd.read_csv('C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/labels/val.csv')

# Define the directory paths
base_dir = "C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Define output directory to save batches
output_dir = os.path.join(base_dir, "batches")
os.makedirs(output_dir, exist_ok=True)

# Convert bounding box data from string format to a list of integers
def parse_bbox(bbox_str):
    return [int(coord) for coord in bbox_str.strip("[]").split(", ")]

# Apply bounding box parsing to the loaded data
train_labels['bbox'] = train_labels['bbox'].apply(parse_bbox)
val_labels['bbox'] = val_labels['bbox'].apply(parse_bbox)

# Define function to check for the file with either .png or .jpg extension
def find_image_path(img_dir, filename):
    img_path_png = os.path.join(img_dir, filename + ".png")
    img_path_jpg = os.path.join(img_dir, filename + ".jpg")
    
    if tf.io.gfile.exists(img_path_png):
        return img_path_png
    elif tf.io.gfile.exists(img_path_jpg):
        return img_path_jpg
    else:
        return None  # Return None if neither exists

# Define function to process images
def load_image(filename, label, bbox, img_dir):
    # Ensure filename is processed as a standard Python string
    filename_no_ext = filename.numpy().decode("utf-8").split(".")[0]
    img_dir = img_dir.numpy().decode("utf-8")
    img_path = find_image_path(img_dir, filename_no_ext)

    if img_path is None:
        print(f"Warning: {filename.numpy().decode()} not found in either .png or .jpg format. Skipping.")
        return tf.zeros([224, 224, 3], dtype=tf.float16), label, bbox  # Return a blank image in float16 if missing

    # Load and preprocess the image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [224, 224])  # Resize to standard input size for model
    img = tf.cast(img, tf.float16) / 255.0  # Normalize and convert to float16
    return img, label, bbox

# Create TensorFlow datasets for train and validation
def create_tf_dataset(labels_df, img_dir):
    filenames = labels_df['filename'].values
    classes = labels_df['class'].values
    bboxes = labels_df['bbox'].tolist()
    
    img_dir_tensor = tf.constant(img_dir)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, classes, bboxes))
    
    dataset = dataset.map(lambda f, c, b: tf.py_function(
        func=load_image, inp=[f, c, b, img_dir_tensor], Tout=(tf.float16, tf.string, tf.int32)))
    
    dataset = dataset.map(lambda img, label, bbox: (img, {'class': label, 'bbox': bbox}))
    dataset = dataset.batch(2000)
    return dataset

# Save each batch
def save_batches(dataset, save_dir, prefix):
    for i, batch in enumerate(dataset):
        batch_dir = os.path.join(save_dir, f"{prefix}_batch_{i}")
        
        # Re-wrap the batch into a tf.data.Dataset object
        batch_dataset = tf.data.Dataset.from_tensor_slices(batch)
        
        # Save the batch dataset
        batch_dataset.save(batch_dir)
        print(f"Saved {prefix} batch {i} to {batch_dir}")


# Create and save datasets for train and validation
train_dataset = create_tf_dataset(train_labels, train_dir)
val_dataset = create_tf_dataset(val_labels, val_dir)

# Save each batch in the specified directory
save_batches(train_dataset, output_dir, "train")
save_batches(val_dataset, output_dir, "val")
