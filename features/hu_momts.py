import cv2
import numpy as np

def extract_hu_moments(image):
    # Convert image to binary (if needed)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Calculate Hu Moments
    moments = cv2.moments(binary)
    hu_moments = cv2.HuMoments(moments).flatten()

    return hu_moments
