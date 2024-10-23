# Load the trained model
from tensorflow.keras.models import load_model

model = load_model('object_classifier_model.h5')

# Preprocess test images similarly to training
test_csv_path = 'path_to_test_csv/test.csv'
test_data = pd.read_csv(test_csv_path)

X_test = []
image_names = []

for index, row in test_data.iterrows():
    image_name = row['Image name']
    bbox = eval(row['Bounding box'])
    
    image_path = os.path.join(image_directory, image_name)
    image = cv2.imread(image_path)
    
    if image is not None:
        processed_image = preprocess_image(image, bbox)
        X_test.append(processed_image)
        image_names.append(image_name)

X_test = np.array(X_test)

# Predict class for each image
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)

# Display predictions
for image_name, predicted_label in zip(image_names, predicted_labels):
    print(f"Image: {image_name}, Predicted Label: {predicted_label}")

