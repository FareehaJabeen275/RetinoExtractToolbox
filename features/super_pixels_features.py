# superpixel_processing.py
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import numpy as np

def extract_superpixels(image, n_segments=100, compactness=10):
    """ Extract superpixels using SLIC (Simple Linear Iterative Clustering) """
    segments = slic(image, n_segments=n_segments, compactness=compactness)
    return segments

def visualize_superpixels(image, segments):
    """ Visualize superpixels """
    plt.imshow(segments, cmap='nipy_spectral')
    plt.title("Superpixels")
    plt.axis('off')
    plt.show()
