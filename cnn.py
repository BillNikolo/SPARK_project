import cv2
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load CSV
train_csv_path = 'path_to_train_csv/train.csv'
train_data = pd.read_csv(train_csv_path)

image_directory = 'path_to_images/'

# Prepare the data
X = []
y = []

def preprocess_image(image, bbox):
    x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
    cropped_image = image[y_min:y_max, x_min:x_max]
    resized_image = cv2.resize(cropped_image, (128, 128))  # Resize to 128x128
    normalized_image = resized_image / 255.0  # Normalize pixel values
    return normalized_image

for index, row in train_data.iterrows():
    image_name = row['Image name']
    bbox = eval(row['Bounding box'])
    label = row['Class']

    image_path = os.path.join(image_directory, image_name)
    image = cv2.imread(image_path)
    
    if image is not None:
        processed_image = preprocess_image(image, bbox)
        X.append(processed_image)
        y.append(label)
    else:
        print(f"Image {image_name} not found.")

# Convert lists to arrays
X = np.array(X)
y = np.array(y)

# Encode labels to integers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer for multi-class classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model
model.save('object_classifier_model.h5')


