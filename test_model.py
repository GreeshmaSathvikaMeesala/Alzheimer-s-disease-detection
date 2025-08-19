import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from alzheimer_detection import EnhancedModel, AdvancedPreprocessor

def test_model():
    # Initialize the model
    model = EnhancedModel()
    
    # Load and preprocess a sample image
    preprocessor = AdvancedPreprocessor()
    
    # Path to your test image
    test_image_path = "test_images/sample.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"Please place a sample brain MRI image at '{test_image_path}'")
        print("The image should be a brain MRI scan in JPG format.")
        return
    
    # Load and preprocess the image
    img = load_img(test_image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    processed_img = preprocessor.preprocess_image(img_array)
    processed_img = np.expand_dims(processed_img, axis=0)
    
    # Get prediction
    prediction = model.predict(processed_img)
    
    # Display results
    plt.figure(figsize=(10, 5))
    
    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Display prediction
    plt.subplot(1, 2, 2)
    plt.bar(['Normal', 'Alzheimer\'s'], [1-prediction[0][0], prediction[0][0]])
    plt.title('Prediction Probability')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\nPrediction Results:")
    print(f"Probability of Alzheimer's: {prediction[0][0]:.2%}")
    print(f"Probability of Normal: {(1-prediction[0][0]):.2%}")
    
    # Generate Grad-CAM visualization
    grad_cam_img = model.grad_cam(test_image_path)
    plt.figure(figsize=(10, 5))
    plt.imshow(grad_cam_img)
    plt.title('Grad-CAM Visualization')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    test_model() 