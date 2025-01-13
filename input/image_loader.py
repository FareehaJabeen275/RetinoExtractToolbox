import cv2
import numpy as np

def load_image(file_path, color_mode='grayscale'):
    if color_mode == 'grayscale':
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    elif color_mode == 'color':
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    else:
        raise ValueError("Unsupported color mode. Use 'grayscale' or 'color'.")

    if image is None:
        raise FileNotFoundError(f"Image not found at {file_path}")
    
    return image

def preprocess_image(image, size=(256, 256)):
    resized_image = cv2.resize(image, size)
    normalized_image = resized_image / 255.0
    return normalized_image
