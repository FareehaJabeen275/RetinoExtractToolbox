import numpy as np
import cv2

def box_count(img, box_size):
    """Count the number of boxes needed to cover the image."""
    img = (img > 0).astype(int)  # Use int instead of np.int
    count = 0
    for i in range(0, img.shape[0], box_size):
        for j in range(0, img.shape[1], box_size):
            if np.any(img[i:i+box_size, j:j+box_size]):
                count += 1
    return count

def fractal_dimension(image):
    """Calculate the fractal dimension of an image using box-counting."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (256, 256))  # Resize to standard size for consistency
    image = (image > 0).astype(int)  # Use int instead of np.int

    sizes = np.arange(2, 20)
    counts = [box_count(image, size) for size in sizes]
    sizes = np.log(sizes)
    counts = np.log(counts)

    coeffs = np.polyfit(sizes, counts, 1)
    return -coeffs[0]

def extract_fractal_dimension(image):
    """Extract fractal dimension and related features."""
    fd = fractal_dimension(image)
    return {
        'fractal_dimension': fd
    }
