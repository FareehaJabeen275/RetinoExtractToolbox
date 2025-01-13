import numpy as np
from scipy.stats import skew, kurtosis
import cv2

def extract_chip_histogram_features(image, chip_coords, bins=256):
    """
    Extracts histogram features from a specified chip (region) of an image.

    Parameters:
    - image: The input image (grayscale or RGB).
    - chip_coords: Coordinates defining the chip (x, y, width, height).
    - bins: Number of bins for the histogram.

    Returns:
    - features: Dictionary of features extracted from the chip histogram.
    """
    x, y, w, h = chip_coords
    chip = image[y:y+h, x:x+w]
    
    # Convert to grayscale if image is RGB
    if len(chip.shape) == 3:
        chip = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
    
    # Compute the histogram
    hist, bin_edges = np.histogram(chip, bins=bins, range=(0, 256))
    
    # Compute histogram features
    mean = np.mean(hist)
    std_dev = np.std(hist)
    variance = np.var(hist)
    skewness = skew(hist)
    kurt = kurtosis(hist)
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    energy = np.sum(hist**2)
    
    features = {
        'Mean': mean,
        'Standard Deviation': std_dev,
        'Variance': variance,
        'Skewness': skewness,
        'Kurtosis': kurt,
        'Entropy': entropy,
        'Energy': energy
    }
    
    return features