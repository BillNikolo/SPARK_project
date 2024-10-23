# Load test CSV
test_csv_path = 'path_to_test_csv/test.csv'
test_data = pd.read_csv(test_csv_path)

# Loop through the test dataset and process each image
for index, row in test_data.iterrows():
    image_name = row['Image name']
    image_path = os.path.join(image_directory, image_name)
    
    # Load the test image
    image = cv2.imread(image_path)
    
    if image is not None:
        # Predict bounding boxes and classify objects
        output_image = predict_bounding_box(image)
        
        # Display or save the output
        output_image_path = os.path.join('test_output_images', image_name)
        cv2.imwrite(output_image_path, output_image)
    else:
        print(f"Test Image {image_name} not found.")
