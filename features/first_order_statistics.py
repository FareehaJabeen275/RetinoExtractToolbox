import numpy as np
from scipy.stats import skew, kurtosis, mode

def extract_first_order_statistics(image):
    """
    Extract first-order statistics from an image.
    
    Parameters:
        image (numpy.ndarray): Input image.
        
    Returns:
        dict: A dictionary with first-order statistics.
    """
    # Flatten the image to a 1D array
    flat_image = image.flatten()
    
    # Calculate mean
    mean = np.mean(flat_image)
    
    # Calculate variance
    variance = np.var(flat_image)
    
    # Calculate standard deviation
    std_dev = np.std(flat_image)
    
    # Calculate skewness
    skewness = skew(flat_image)
    
    # Calculate kurtosis
    kurt = kurtosis(flat_image)
    
    # Calculate entropy
    entropy_val = entropy(flat_image)
    
    # Calculate energy
    energy = np.sum(flat_image ** 2)
    
    # Calculate maximum
    max_val = np.max(flat_image)
    
    # Calculate minimum
    min_val = np.min(flat_image)
    
    # Calculate range
    range_val = max_val - min_val
    
    # Calculate median
    median_val = np.median(flat_image)
    
   
    # Return features as a dictionary
    return {
        'mean': mean,
        'variance': variance,
        'standard_deviation': std_dev,
        'skewness': skewness,
        'kurtosis': kurt,
        'entropy': entropy_val,
        'energy': energy,
        'maximum': max_val,
        'minimum': min_val,
        'range': range_val,
        'median': median_val
    }

def entropy(image):
    """
    Compute the entropy of the image.
    
    Parameters:
        image (numpy.ndarray): Input image.
    
    Returns:
        float: Entropy value.
    """
    # Compute histogram
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # Normalize histogram
    
    # Compute entropy
    hist = hist[hist > 0]  # Avoid log(0)
    return -np.sum(hist * np.log(hist))
