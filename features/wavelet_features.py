import pywt
import numpy as np
import scipy.stats as stats

def extract_wavelet_features(image, wavelet='db1', level=2):
    """
    Extract wavelet features from an image and additional statistical features from wavelet coefficients.
    
    Parameters:
        image (numpy.ndarray): Input image.
        wavelet (str): Wavelet type (e.g., 'db1', 'haar', 'sym2').
        level (int): Decomposition level.
        
    Returns:
        dict: A dictionary with wavelet and statistical features.
    """
    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # Extract features from coefficients
    features = {}
    for i, coeff in enumerate(coeffs):
        if isinstance(coeff, tuple):
            for j, sub_coeff in enumerate(coeff):
                flat = sub_coeff.flatten()
                # Compute statistics for each sub-band
                features[f'coeff_{i}_{j}_mean'] = np.mean(sub_coeff)
                features[f'coeff_{i}_{j}_variance'] = np.var(sub_coeff)
                features[f'coeff_{i}_{j}_energy'] = np.sum(sub_coeff ** 2)
                features[f'coeff_{i}_{j}_entropy'] = entropy(sub_coeff)
                features[f'coeff_{i}_{j}_michelson_contrast'] = michelson_contrast(sub_coeff)
                
                if np.var(flat) > 1e-10:
                    features[f'coeff_{i}_{j}_skewness'] = stats.skew(flat)
                    features[f'coeff_{i}_{j}_kurtosis'] = stats.kurtosis(flat)
                else:
                    features[f'coeff_{i}_{j}_skewness'] = 0.0
                    features[f'coeff_{i}_{j}_kurtosis'] = 0.0
                
                features[f'coeff_{i}_{j}_min'] = np.min(sub_coeff)
                features[f'coeff_{i}_{j}_max'] = np.max(sub_coeff)
                features[f'coeff_{i}_{j}_std_dev'] = np.std(sub_coeff)
                features[f'coeff_{i}_{j}_mad'] = np.mean(np.abs(sub_coeff - np.mean(sub_coeff)))
        else:
            flat = coeff.flatten()
            # Compute statistics for each band
            features[f'coeff_{i}_mean'] = np.mean(coeff)
            features[f'coeff_{i}_variance'] = np.var(coeff)
            features[f'coeff_{i}_energy'] = np.sum(coeff ** 2)
            features[f'coeff_{i}_entropy'] = entropy(coeff)
            features[f'coeff_{i}_michelson_contrast'] = michelson_contrast(coeff)
            
            if np.var(flat) > 1e-10:
                features[f'coeff_{i}_skewness'] = stats.skew(flat)
                features[f'coeff_{i}_kurtosis'] = stats.kurtosis(flat)
            else:
                features[f'coeff_{i}_skewness'] = 0.0
                features[f'coeff_{i}_kurtosis'] = 0.0
                
            features[f'coeff_{i}_min'] = np.min(coeff)
            features[f'coeff_{i}_max'] = np.max(coeff)
            features[f'coeff_{i}_std_dev'] = np.std(coeff)
            features[f'coeff_{i}_mad'] = np.mean(np.abs(coeff - np.mean(coeff)))
    
    return features

def michelson_contrast(coeff):
    """
    Compute the Michelson contrast of the coefficient matrix.
    
    Parameters:
        coeff (numpy.ndarray): Wavelet coefficients.
    
    Returns:
        float: Michelson contrast value.
    """
    I_max = np.max(coeff)
    I_min = np.min(coeff)
    
    if (I_max + I_min) == 0:
        return 0  # To avoid division by zero
    
    return (I_max - I_min) / (I_max + I_min)

def entropy(coeff):
    """
    Compute the entropy of the coefficient matrix.
    
    Parameters:
        coeff (numpy.ndarray): Wavelet coefficients.
    
    Returns:
        float: Entropy value.
    """
    # Normalize coefficients safely
    coeff = coeff - np.min(coeff)
    max_val = np.max(coeff)
    if max_val > 0:
        coeff = coeff / max_val
    else:
        # All zeros, entropy = 0
        return 0.0
    
    # Compute histogram
    hist, _ = np.histogram(coeff, bins=256, range=(0, 1))
    total = hist.sum()
    if total > 0:
        hist = hist / total
    else:
        return 0.0
    
    # Compute entropy
    hist = hist[hist > 0]  # Avoid log(0)
    return -np.sum(hist * np.log(hist))
