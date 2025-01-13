import cv2
import mahotas
import numpy as np
def extract_zernike_moments(image, radius=21):
    # Ensure the image is grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding to create a binary image
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    zernike_moments = mahotas.features.zernike_moments(image, radius)
 
    
    return zernike_moments
