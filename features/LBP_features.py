import numpy as np
from skimage import color
from skimage.feature import local_binary_pattern
from scipy.stats import skew, kurtosis
import argparse
import matplotlib.pyplot as plt

def extract_lbp_features(image, P=8, R=1, method='uniform'):
    """
    Extracts LBP features from the given image using the specified method.

    Parameters:
    - image: The input image (grayscale).
    - P: Number of circularly symmetric neighbor set points (default is 8).
    - R: Radius of the circle (default is 1).
    - method: Method for LBP computation ('uniform', 'nri_uniform', 'ror', 'var').

    Returns:
    - features: Dictionary of features extracted from the LBP histogram.
    """
    # Convert to grayscale if the image is RGB
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    
    # Compute LBP image
    lbp = local_binary_pattern(image, P=P, R=R, method=method)
    
    # Compute histogram of LBP values
    n_bins = P + 2  # Number of bins is P + 2 for uniform patterns
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    
    # Normalize histogram
    hist = hist.astype('float')
    hist /= hist.sum()
    
    # Compute statistical features from histogram
    mean = np.mean(hist)
    std_dev = np.std(hist)
    variance = np.var(hist)
    skewness = skew(hist)
    kurt = kurtosis(hist)
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    
    features = {
        'Mean': mean,
        'Standard Deviation': std_dev,
        'Variance': variance,
        'Skewness': skewness,
        'Kurtosis': kurt,
        'Entropy': entropy
    }
    
    return features

