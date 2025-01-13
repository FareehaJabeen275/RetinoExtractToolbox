
import pandas as pd
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from features.glcm_features import extract_glcm_features
from features.wavelet_features import extract_wavelet_features
from features.image_statistics import extract_image_statistics
from features.histogram_based_statistics import extract_histogram_features
from features.gabor_features import extract_gabor_features
from features.chip_histogram_features import extract_chip_histogram_features
from features.LBP_features import extract_lbp_features
from features.HOG_features import extract_hog_features
from features.fractal_dimension import extract_fractal_dimension
from features.superpixel_processing import extract_superpixels, visualize_superpixels, compute_superpixel_features
from features.glrlm_features import extract_glrlm_features
from features.zernike_momts import extract_zernike_moments
from features.hu_momts import extract_hu_moments
from features.moments_all import compute_moments
from input.image_loader import load_image, preprocess_image
from skimage.color import rgb2hsv

# def save_features_to_csv(features, output_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for feature_name, feature_data in features.items():
#         file_path = os.path.join(output_dir, f'{feature_name}.csv')
        
#         if isinstance(feature_data, dict):
#             # Convert dictionary to DataFrame with proper column names
#             df = pd.DataFrame(list(feature_data.items()), columns=['Feature', 'Value'])
#         elif isinstance(feature_data, list):
#             if isinstance(feature_data[0], (list, tuple)):
#                 # If list contains lists or tuples, create DataFrame directly
#                 df = pd.DataFrame(feature_data)
#             else:
#                 # Otherwise, assume list of values
#                 df = pd.DataFrame(feature_data, columns=['Value'])
#         else:
#             # Handle single values
#             df = pd.DataFrame([feature_data], columns=['Value'])
        
#         # Save DataFrame to CSV
#         df.to_csv(file_path, index=False)
#         print(f"Saved {feature_name} features to {file_path}")
def save_features_to_csv(features, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for feature_name, feature_data in features.items():
        file_path = os.path.join(output_dir, f'{feature_name}.csv')
        
        # Check the type of feature_data and handle accordingly
        if isinstance(feature_data, dict):
            # Convert dictionary to DataFrame with proper column names
            df = pd.DataFrame(list(feature_data.items()), columns=['Feature', 'Value'])
        elif isinstance(feature_data, list):
            if isinstance(feature_data[0], (list, tuple)):
                # If list contains lists or tuples, create DataFrame directly
                df = pd.DataFrame(feature_data)
            else:
                # Assuming feature_data is a list of values
                df = pd.DataFrame(feature_data, columns=['Value'])
        elif isinstance(feature_data, (np.ndarray, np.generic)):
            # Convert numpy array to DataFrame
            df = pd.DataFrame(feature_data, columns=['Value'])
        else:
            # Handle single values
            df = pd.DataFrame([feature_data], columns=['Value'])
        
        # Save DataFrame to CSV
        df.to_csv(file_path, index=False)
        print(f"Saved {feature_name} features to {file_path}")


def main(image_path, wavelet, level, lbp_P, lbp_R, lbp_method,
         hog_pixels_per_cell, hog_cells_per_block, n_segments, compactness, radius, feature_flags, output_dir):
    # Load and preprocess the image
    image = load_image(image_path, color_mode='grayscale')

    preprocessed_image = preprocess_image(image)
    
    image1 = load_image(image_path, color_mode='color')
    
    extracted_features = {}

    if feature_flags.get('superpixel', False):
        original_image, segments, segmented_image = extract_superpixels(image1, n_segments=n_segments, compactness=compactness)
        visualize_superpixels(original_image, segments, segmented_image)
        superpixel_features = compute_superpixel_features(image1, segments)
        extracted_features['superpixel_features'] = superpixel_features
    # Convert the preprocessed image back to uint8 format
    preprocessed_image_uint8 = (preprocessed_image * 255).astype(np.uint8)
    
    if feature_flags.get('moments', False):
        moments = compute_moments(preprocessed_image_uint8)
        extracted_features['moments'] = moments
    
    if feature_flags.get('glcm', False):
        glcm_features = extract_glcm_features(preprocessed_image_uint8)
        extracted_features['glcm'] = glcm_features

    if feature_flags.get('glrlm', False):
        glrlm_features = extract_glrlm_features(preprocessed_image_uint8)
        extracted_features['glrlm'] = glrlm_features

    if feature_flags.get('wavelet', False):
        wavelet_features = extract_wavelet_features(preprocessed_image_uint8, wavelet=wavelet, level=level)
        extracted_features['wavelet'] = wavelet_features

    if feature_flags.get('image_stats', False):
        statistical_features = extract_image_statistics(preprocessed_image_uint8)
        extracted_features['image_stats'] = statistical_features

    if feature_flags.get('histogram', False):
        histogram_based_statistics = extract_histogram_features(preprocessed_image_uint8)
        extracted_features['histogram'] = histogram_based_statistics

    if feature_flags.get('gabor', False):
        gabor_features = extract_gabor_features(preprocessed_image_uint8)
        extracted_features['gabor'] = gabor_features

    if feature_flags.get('chip_histogram', False):
        chip_coords = (50, 50, 100, 100)  # Example coordinates
        chip_features = extract_chip_histogram_features(image, chip_coords)
        extracted_features['chip_histogram'] = chip_features

    if feature_flags.get('lbp', False):
        lbp_features = extract_lbp_features(preprocessed_image_uint8, P=lbp_P, R=lbp_R, method=lbp_method)
        extracted_features['lbp'] = lbp_features

    if feature_flags.get('hog', False):
        hog_features, hog_image_rescaled = extract_hog_features(preprocessed_image_uint8, 
                                            pixels_per_cell=tuple(hog_pixels_per_cell),
                                            cells_per_block=tuple(hog_cells_per_block), 
                                            visualize=True)
        extracted_features['hog'] = hog_features

    if feature_flags.get('fractal', False):
        fractal_features = extract_fractal_dimension(preprocessed_image_uint8)
        extracted_features['fractal'] = fractal_features

    if feature_flags.get('hu_moments', False):
        hu_moments = extract_hu_moments(preprocessed_image_uint8)
        extracted_features['hu_moments'] = hu_moments

    if feature_flags.get('zernike', False):
        zernike_moments = extract_zernike_moments(preprocessed_image_uint8, radius=radius)
        extracted_features['zernike'] = zernike_moments



        # Display the extracted features
    for feature_name, features in extracted_features.items():
        print(f"\n{feature_name.capitalize()} Features:")
        if isinstance(features, dict):
            for key, value in features.items():
                print(f"{key}: {value}")
        else:
            print(features)

    # Save extracted features to CSV files
    save_features_to_csv(extracted_features, output_dir)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process and display a medical image.')
    parser.add_argument('image_path', type=str, help='Path to the input image file')
    parser.add_argument('--wavelet_type', type=str, default='db1', help='Type of wavelet to use (default: db1)')
    parser.add_argument('--level', type=int, default=2, help='Level of wavelet decomposition (default: 2)')
    parser.add_argument('--lbp_P', type=int, default=8, help='Number of points for LBP (default: 8)')
    parser.add_argument('--lbp_R', type=int, default=1, help='Radius for LBP (default: 1)')
    parser.add_argument('--lbp_method', type=str, default='uniform', help='Method for LBP (default: uniform)')
    parser.add_argument('--hog_pixels_per_cell', type=int, nargs=2, default=[8, 8], help='Pixels per cell for HOG (default: [8, 8])')
    parser.add_argument('--hog_cells_per_block', type=int, nargs=2, default=[2, 2], help='Cells per block for HOG (default: [2, 2])')
    parser.add_argument('--n_segments', type=int, default=100, help='Number of segments for superpixel (default: 100)')
    parser.add_argument('--compactness', type=int, default=10, help='Value of compactness for superpixel (default: 10)')
    parser.add_argument('--radius', type=int, default=8, help='Value of radius for Zernike moment calculation (default: 8)')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save extracted features')

    # Feature extraction flags
    parser.add_argument('--glcm', action='store_true', help='Extract GLCM features')
    parser.add_argument('--glrlm', action='store_true', help='Extract GLRLM features')
    parser.add_argument('--wavelet', action='store_true', help='Extract Wavelet features')
    parser.add_argument('--image_stats', action='store_true', help='Extract Image Statistics')
    parser.add_argument('--histogram', action='store_true', help='Extract Histogram based Statistics')
    parser.add_argument('--gabor', action='store_true', help='Extract Gabor features')
    parser.add_argument('--chip_histogram', action='store_true', help='Extract Chip Histogram features')
    parser.add_argument('--lbp', action='store_true', help='Extract LBP features')
    parser.add_argument('--hog', action='store_true', help='Extract HOG features')
    parser.add_argument('--fractal', action='store_true', help='Extract Fractal Dimension features')
    parser.add_argument('--moments', action='store_true', help='Extract moments')
    parser.add_argument('--hu_moments', action='store_true', help='Extract Hu Moments')
    parser.add_argument('--zernike', action='store_true', help='Extract Zernike Moments')
    parser.add_argument('--superpixel', action='store_true', help='Extract Superpixel features')

    args = parser.parse_args()
    
    # Prepare feature flags
    feature_flags = {
        'glcm': args.glcm,
        'glrlm': args.glrlm,
        'wavelet': args.wavelet,
        'image_stats': args.image_stats,
        'histogram': args.histogram,
        'gabor': args.gabor,
        'chip_histogram': args.chip_histogram,
        'lbp': args.lbp,
        'hog': args.hog,
        'fractal': args.fractal,
        'moments' : args.moments,
        'hu_moments': args.hu_moments,
        'zernike': args.zernike,
        'superpixel': args.superpixel
    }

    # Run main function
    main(args.image_path, args.wavelet_type, args.level, args.lbp_P, args.lbp_R, args.lbp_method,
         args.hog_pixels_per_cell, args.hog_cells_per_block, args.n_segments, args.compactness, args.radius,
         feature_flags, args.output_dir)
