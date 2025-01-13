# features/preprocessing.py
import cv2
import numpy as np
from skimage import exposure

def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)

def normalize_image(image):
    # Normalize image to range [0, 1]
    if image.dtype == np.uint8:
        image = image / 255.0
    elif image.dtype == np.float64:
        image = np.clip(image, 0, 1)  # Ensure values are in the range [0, 1]
    return image
# def normalize_image(image):
#     # Ensure the image is in float format
#     if image.dtype != np.float32:
#         image = image.astype(np.float32)
    
#     # Normalize the image to range [0, 1]
#     normalized_image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
    
#     # Scale to range [0, 255] for 8-bit display
#     normalized_image = (normalized_image * 255).astype(np.uint8)
    
#     return normalized_image

def equalize_histogram(image):
    return cv2.equalizeHist(image)

def adaptive_histogram_equalization(image):
    # Ensure image is in float format and normalized
    if image.dtype == np.uint8:
        image = image / 255.0
    # Apply adaptive histogram equalization
    equalized_image = exposure.equalize_adapthist(image)
    # Convert back to uint8 if needed
    return (equalized_image * 255).astype(np.uint8)

def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def gaussian_blur(image, ksize=(5, 5)):
    return cv2.GaussianBlur(image, ksize, 0)

def contrast_stretch(image, min_out=0, max_out=255):
    p2, p98 = np.percentile(image, (2, 98))
    return exposure.rescale_intensity(image, in_range=(p2, p98), out_range=(min_out, max_out))

def median_blur(image, ksize=5):
    return cv2.medianBlur(image, ksize)

def bilateral_filtering(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def morphological_operations(image, operation='dilation', kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'dilation':
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == 'erosion':
        return cv2.erode(image, kernel, iterations=1)
    elif operation == 'opening':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def extract_color_channels(image, color_space='RGB'):
    if color_space == 'RGB':
        return cv2.split(image)
    elif color_space == 'HSV':
        return cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    elif color_space == 'LAB':
        return cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2Lab))


def extract_roi(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)
