# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import io, color
# from skimage.segmentation import slic
# from skimage.color import label2rgb

# def extract_superpixels(image, n_segments=100, compactness=10):
#     """ Perform SLIC superpixel segmentation on the input image. """
    
#     # Apply SLIC
#     segments = slic(image, n_segments=n_segments, compactness=compactness, start_label=1)
    
#     # Convert segmented image to RGB
#     segmented_image = label2rgb(segments, image, kind='avg')
    
#     return image, segments, segmented_image

# def visualize_superpixels(original_image, segments, segmented_image):
#     """ Visualize the original image, superpixel segments, and segmented image. """
#     fig, ax = plt.subplots(1, 3, figsize=(15, 5))

#     ax[0].imshow(original_image)
#     ax[0].set_title('Original Image')
#     ax[0].axis('off')

#     ax[1].imshow(segments, cmap='tab20b')
#     ax[1].set_title('Superpixel Segments')
#     ax[1].axis('off')

#     ax[2].imshow(segmented_image)
#     ax[2].set_title('Segmented Image')
#     ax[2].axis('off')

#     plt.show()
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.util import img_as_float
from skimage.filters import threshold_otsu
from skimage import exposure

def extract_superpixels(image, n_segments=100, compactness=10):
    """ Perform SLIC superpixel segmentation on the input image. """
    
    # Apply SLIC
    segments = slic(image, n_segments=n_segments, compactness=compactness, start_label=1)
    
    # Convert segmented image to RGB
    segmented_image = label2rgb(segments, image, kind='avg')
    
    # Count unique superpixels
    unique_segments = np.unique(segments)
    num_superpixels = len(unique_segments)
    
    print(f"Number of superpixels created: {num_superpixels}")
    return image, segments, segmented_image

def compute_superpixel_features(image, segments):
    """ Compute features for each superpixel region. """
    
    # Convert image to float
    image_float = img_as_float(image)
    
    # Extract properties for each superpixel
    props = regionprops(segments, intensity_image=image_float)
    
    features = []
    for region in props:
        # Extract intensity values for the current superpixel
        intensity_values = image_float[segments == region.label]
        
        # Compute features
        mean_intensity = np.mean(intensity_values)
        std_intensity = np.std(intensity_values)
        min_intensity = np.min(intensity_values)
        max_intensity = np.max(intensity_values)
        
        # Histogram statistics
        hist, _ = np.histogram(intensity_values, bins=10, range=(0, 1))
        hist_stats = {
            'hist_mean': np.mean(hist),
            'hist_std': np.std(hist),
            'hist_max': np.max(hist),
            'hist_min': np.min(hist)
        }
        
        # Append features for the current superpixel
        features.append({
            'label': region.label,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'min_intensity': min_intensity,
            'max_intensity': max_intensity,
            **hist_stats
        })
    
    return features

def visualize_superpixels(original_image, segments, segmented_image):
    """ Visualize the original image, superpixel segments, and segmented image. """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(segments, cmap='tab20b')
    ax[1].set_title('Superpixel Segments')
    ax[1].axis('off')

    ax[2].imshow(segmented_image)
    ax[2].set_title('Segmented Image')
    ax[2].axis('off')

    plt.show()

