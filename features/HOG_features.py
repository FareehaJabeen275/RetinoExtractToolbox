from skimage.feature import hog
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

def extract_hog_features(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True):
    """
    Extracts HOG features from the given image and calculates statistical features.

    Parameters:
    - image: The input image (grayscale).
    - pixels_per_cell: Size of each cell in pixels (tuple of two integers).
    - cells_per_block: Number of cells per block (tuple of two integers).
    - visualize: If True, return an image with HOG visualization.

    Returns:
    - stats: Dictionary with statistical features.
    - hog_image: The HOG visualization image if visualize is True.
    """
    # Validate parameters
    if not isinstance(pixels_per_cell, tuple) or not isinstance(cells_per_block, tuple):
        raise ValueError("pixels_per_cell and cells_per_block must be tuples of integers.")
    
    if len(pixels_per_cell) != 2 or len(cells_per_block) != 2:
        raise ValueError("pixels_per_cell and cells_per_block must each be tuples of length 2.")
    
    if any(type(val) is not int or val <= 0 for val in pixels_per_cell + cells_per_block):
        raise ValueError("All values in pixels_per_cell and cells_per_block must be positive integers.")
    
    if image.ndim != 2:
        raise ValueError("Image must be grayscale.")
    
    # Extract HOG features
    if visualize:
        features, hog_image = hog(image, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                                  visualize=True, block_norm='L2-Hys')
        # Scale the HOG image for better visualization
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        
        # Display HOG image
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(hog_image_rescaled, cmap='gray')
        plt.title('HOG Visualization')
        plt.axis('off')
        plt.show()
    else:
        features = hog(image, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                       visualize=False, block_norm='L2-Hys')
    
    # Compute statistical features
    mean = np.mean(features)
    variance = np.var(features)
    std_dev = np.std(features)
    kurt = kurtosis(features)
    energy = np.sum(features ** 2)
    
    stats = {
        'feature_length': len(features),
        'mean': mean,
        'variance': variance,
        'standard_deviation': std_dev,
        'kurtosis': kurt,
        'energy': energy
    }
    
    if visualize:
        return stats, hog_image_rescaled
    else:
        return stats
