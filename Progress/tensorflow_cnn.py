import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, Model
from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision
set_global_policy('mixed_float16')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define class names from CSV
class_names = [
    'smart_1', 'cheops', 'lisa_pathfinder', 'debris', 'proba_3_ocs',
    'proba_3_csc', 'soho', 'earth_observation_sat_1', 'proba_2',
    'xmm_newton', 'double_star'
]

# Parameters
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_CLASSES = len(class_names)  # Set to 11 based on class names
BATCH_COUNT = 5  # Adjust this to control the number of batches to load
BATCH_DIR = "C:/Users/vniko/Desktop/CVIA/spark-2022-stream-1/batches"

# Data Generator Class for Loading Saved Batches
class BatchDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_dir, batch_prefix="train", batch_count=BATCH_COUNT, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
        self.batch_dir = batch_dir
        self.batch_prefix = batch_prefix
        self.batch_count = batch_count
        self.batch_size = batch_size
        self.img_size = img_size
        self.class_names = class_names
        self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
        self.loaded_batches = self.load_batches()

    def load_batches(self):
        batches = []
        for i in range(self.batch_count):
            batch_path = os.path.join(self.batch_dir, f"{self.batch_prefix}_batch_{i}")
            if os.path.exists(batch_path):
                batch = tf.data.Dataset.load(batch_path)
                batches.append(batch)
            else:
                print(f"Batch {i} not found at {batch_path}")
        return batches

    def __len__(self):
        return sum([len(batch) for batch in self.loaded_batches]) // self.batch_size

    def __getitem__(self, idx):
        batch_data = self.loaded_batches[idx % self.batch_count]
        batch_images, batch_bboxes, batch_classes = [], [], []
        
        for img, labels in batch_data.take(self.batch_size):
            # Resize to 256x256
            img = tf.image.resize(img, (self.img_size, self.img_size))
            batch_images.append(img.numpy())
            batch_bboxes.append(labels['bbox'].numpy())

            # Convert string labels to numeric, handle missing labels
            class_label = labels['class'].numpy().decode("utf-8")
            class_index = self.class_mapping.get(class_label)
            if class_index is None:
                print(f"Warning: Class label '{class_label}' not found in class mapping. Skipping this item.")
                continue
            
            class_one_hot = tf.keras.utils.to_categorical(class_index, num_classes=NUM_CLASSES)
            batch_classes.append(class_one_hot)

        return np.array(batch_images), {"bbox": np.array(batch_bboxes), "class": np.array(batch_classes)}

# Create training and validation generators
train_gen = BatchDataGenerator(batch_dir=BATCH_DIR, batch_prefix="train", batch_count=BATCH_COUNT)
val_gen = BatchDataGenerator(batch_dir=BATCH_DIR, batch_prefix="val", batch_count=BATCH_COUNT)

# Define the model with ResNet50 as the backbone
def create_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    # Backbone network without dtype specification
    backbone = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    backbone.trainable = False  # Freeze the backbone

    # Flatten the backbone output
    x = layers.Flatten()(backbone.output)
    
    # Bounding box regression head
    bbox_head = layers.Dense(1024, activation="relu")(x)
    bbox_head = layers.Dense(512, activation="relu")(bbox_head)
    bbox_output = layers.Dense(4, activation="sigmoid", name="bbox")(bbox_head)
    
    # Classification head
    class_head = layers.Dense(1024, activation="relu")(x)
    class_head = layers.Dense(512, activation="relu")(class_head)
    class_output = layers.Dense(num_classes, activation="softmax", name="class")(class_head)
    
    # Define the model with two outputs
    model = Model(inputs=backbone.input, outputs=[bbox_output, class_output])
    return model

# Initialize and compile the model
model = create_model()
model.compile(
    optimizer="adam",
    loss={
        "bbox": "mse",
        "class": "categorical_crossentropy"
    },
    metrics={"bbox": "mae", "class": "accuracy"}
)

# Train the model with validation data
EPOCHS = 10
model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    verbose=1
)

# Evaluate the model
evaluation = model.evaluate(val_gen)
print("Evaluation Results:", evaluation)
