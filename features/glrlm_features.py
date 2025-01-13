import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glrlm_features(image, num_levels=256, angles=[0], distances=[1]):
    # Ensure the image is in the correct format (e.g., grayscale and uint8)
    if len(image.shape) == 3:
        image = image[:, :, 0]  # Convert to grayscale if needed
    image = (image * (num_levels - 1)).astype(np.uint8)  # Normalize image

    # Compute the GLRLM matrix
    glrlm = graycomatrix(image, distances=distances, angles=angles, levels=num_levels, symmetric=True, normed=True)
    
    # Calculate the GLRLM properties
    SRE = np.sum((glrlm / np.power(np.arange(1, glrlm.shape[0] + 1), 2).reshape(-1, 1, 1, 1)), axis=(0, 1, 2)).flatten()[0]
    LRE = np.sum(glrlm * np.power(np.arange(1, glrlm.shape[0] + 1).reshape(-1, 1, 1, 1), 2), axis=(0, 1, 2)).flatten()[0]
    GLN = np.sum(np.power(np.sum(glrlm, axis=0), 2)).flatten()[0]
    RLN = np.sum(np.power(np.sum(glrlm, axis=1), 2)).flatten()[0]
    LGRE = np.sum(glrlm / np.power(np.arange(1, glrlm.shape[0] + 1).reshape(-1, 1, 1, 1), 2), axis=(0, 1, 2)).flatten()[0]
    HGRE = np.sum(glrlm * np.power(np.arange(1, glrlm.shape[0] + 1).reshape(-1, 1, 1, 1), 2), axis=(0, 1, 2)).flatten()[0]

    features = {
        'SRE': SRE,
        'LRE': LRE,
        'GLN': GLN,
        'RLN': RLN,
        'LGRE': LGRE,
        'HGRE': HGRE
    }
    
    return features
