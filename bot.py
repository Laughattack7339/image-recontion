import numpy as np
import cv2
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os

def load_and_preprocess_image(img_path):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to load image at {img_path}")
        return None
    
    # Resize image to 224x224 pixels
    img = cv2.resize(img, (224, 224))
    
    # Convert image to array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to create a batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the input image
    img_array = preprocess_input(img_array)
    
    return img_array

def main():
    # Load the pre-trained ResNet50 model
    try:
        model = ResNet50(weights='imagenet')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading ResNet50 model: {e}")
        return

    # Path to the image
    img_path = 'dog.jpg'

    # Check if image path exists
    if not os.path.exists(img_path):
        print(f"Error: The image path '{img_path}' does not exist.")
        return

    # Load and preprocess the image
    img_array = load_and_preprocess_image(img_path)
    if img_array is None:
        return

    # Predict the class of the image
    try:
        preds = model.predict(img_array)
        print("Prediction made successfully.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # Decode and print the top-3 predicted classes
    try:
        decoded_preds = decode_predictions(preds, top=3)[0]
        print('Predicted:', decoded_preds)
    except Exception as e:
        print(f"Error decoding predictions: {e}")
        return

    # Display the image (optional)
    cv2.imshow('Image', cv2.imread(img_path))
    print("Press any key to close the image window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
