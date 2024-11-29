from pathlib import Path

import cv2
import os
import torch
from torchvision import transforms
from PIL import Image

from main import LicensePlateRecognizer

# Character mapping (used during training)
int_to_char = {idx: char for idx, char in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")}

# Preprocessing function (same as during training)
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Load the trained model
def load_model(model_path, num_classes=36):
    model = LicensePlateRecognizer(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Set the model to evaluation mode
    return model

# Predict a single character
def predict_character(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return int_to_char[predicted.item()]

# Segment characters from a license plate
def segment_characters(license_plate_path, output_path="segmented_with_contours.jpg"):
    """
    Segment characters from a license plate image, draw contours on the original image,
    and save each segmented character as an image.
    """
    # Load the image
    image = cv2.imread(license_plate_path)
    original_image = image.copy()  # Keep a copy of the original image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)  # Thresholding

    # Detect contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    characters = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 6 and h > 6:  # Filter out noise
            # Draw a bounding box around each character on the original image
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract the character region
            char = extract_character_with_padding(gray, x, y, w, h)

            char = cv2.resize(char, (28, 28))  # Resize to match training
            characters.append(char)

    # Save the image with drawn contours
    cv2.imwrite(output_path, original_image)
    print(f"Image with contours saved to {output_path}")

    return characters
def extract_character_with_padding(gray, x, y, w, h, padding=5):
    """
    Extract a character from the image with padding, ensuring it doesn't exceed the image boundaries.

    Args:
        gray (ndarray): Grayscale input image.
        x, y, w, h (int): Bounding box coordinates and dimensions.
        padding (int): Amount of padding to add around the character.

    Returns:
        ndarray: Padded character region.
    """
    height, width = gray.shape

    # Calculate padded coordinates
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(width, x + w + padding)
    y_end = min(height, y + h + padding)

    # Extract the padded region
    return gray[y_start:y_end, x_start:x_end]
# Predict the full license plate number
def predict_license_plate(model, license_plate_path, char_output_dir="segmented_characters"):
    """
    Predict the full license plate from an image of the license plate.
    """
    # Segment characters from the license plate
    segmented_characters = segment_characters(license_plate_path)

    # Create output directory for segmented characters
    os.makedirs(char_output_dir, exist_ok=True)

    # Predict each character
    index = 0
    license_plate = ""
    for char_image in segmented_characters:
        char_image_pil = Image.fromarray(char_image)  # Convert to PIL Image
        image_tensor = preprocess_image(char_image_pil)
        predicted_char = predict_character(model, image_tensor)
        # Save the character image
        char_path = os.path.join(char_output_dir, f"char_{index}_{predicted_char}.png")
        cv2.imwrite(char_path, char_image)
        print(f"Saved character image: {char_path}")
        index = index+1
        license_plate += predicted_char

    return license_plate

# Example Usage
if __name__ == "__main__":
    # Paths
    model_path =  f"{Path.home()}/hellotorch/dataset/trained_model.pth"
    license_plate_image =  f"{Path.home()}/hellotorch/dataset/license_plate_dataset/plate_4.png"

    # Load the trained model
    model = load_model(model_path)

    # Predict the license plate number
    predicted_plate = predict_license_plate(model, license_plate_image)
    print(f"Predicted License Plate: {predicted_plate}")