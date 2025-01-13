from skimage.feature import graycomatrix, graycoprops
import numpy as np

def extract_haralick_features(image, distances=[1], angles=[0], num_levels=256):
    # Ensure the image is grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute GLCM
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=num_levels, symmetric=True, normed=True)

    # Extract Haralick features
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    ASM = graycoprops(glcm, 'ASM').flatten()

    return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, ASM])
