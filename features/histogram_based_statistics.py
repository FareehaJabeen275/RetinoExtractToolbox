import cv2
import numpy as np
from scipy.stats import skew, kurtosis, entropy

def extract_histogram_features(image):
    # Calculate histogram
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 256), density=True)
    
    # Calculate mean from histogram
    mean_val = np.sum(histogram * bin_edges[:-1])

    # Calculate variance from histogram
    variance_val = np.sum(histogram * (bin_edges[:-1] - mean_val) ** 2)

    # Calculate skewness from histogram
    skewness_val = np.sum(histogram * (bin_edges[:-1] - mean_val) ** 3) / (np.sqrt(variance_val) ** 3)

    # Calculate kurtosis from histogram
    kurtosis_val = np.sum(histogram * (bin_edges[:-1] - mean_val) ** 4) / (variance_val ** 2)

    # Calculate entropy from histogram
    entropy_val = entropy(histogram + np.finfo(float).eps)  # Adding epsilon to avoid log(0)

    # Calculate energy (angular second moment) from histogram
    energy_val = np.sum(histogram ** 2)

    # Calculate contrast (second moment about mean) from histogram
    contrast_val = np.sum((bin_edges[:-1] - mean_val) ** 2 * histogram)

    # Find peaks in the histogram
    peaks = bin_edges[np.where(histogram == np.max(histogram))]

    # Calculate histogram spread (range)
    spread_val = bin_edges[np.max(np.where(histogram > 0))] - bin_edges[np.min(np.where(histogram > 0))]

    # Calculate mode (most frequent bin)
    mode_val = bin_edges[np.argmax(histogram)]

    # Calculate median from cumulative histogram
    cumulative_histogram = np.cumsum(histogram)
    median_idx = np.where(cumulative_histogram >= 0.5)[0][0]
    median_val = bin_edges[median_idx]

    features = {
        'mean': mean_val,
        'variance': variance_val,
        'skewness': skewness_val,
        'kurtosis': kurtosis_val,
        'entropy': entropy_val,
        'energy': energy_val,
        'contrast': contrast_val,
        'peaks': peaks,
        'spread': spread_val,
        'mode': mode_val,
        'median': median_val
    }

    return features