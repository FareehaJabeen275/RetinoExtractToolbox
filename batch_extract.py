import os
import argparse
import pandas as pd
from tqdm import tqdm
from main import main as extract_features_single_image
from main import save_features_to_csv  # optional if you want per-image CSVs
from input.image_loader import load_image  # to validate image loading
import matplotlib
matplotlib.use('Agg')  # disables all plots (no windows will pop up)
import matplotlib.pyplot as plt
import warnings
plt.show = lambda *args, **kwargs: None  # disables all plt.show() calls
warnings.filterwarnings("ignore", message="Matplotlib is currently using agg.*")

def get_all_image_paths(input_dir):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    return [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]

def flatten_features(feature_dict):
    flat = {}
    for key, value in feature_dict.items():
        if isinstance(value, dict):
            for subkey, subval in value.items():
                flat[f"{key}_{subkey}"] = subval
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                flat[f"{key}_{i}"] = item
        else:
            flat[key] = value
    return flat

def batch_extract(input_dir, output_csv, args):
    image_paths = get_all_image_paths(input_dir)
    all_data = []

    print(f"Processing {len(image_paths)} images...")

    for img_path in tqdm(image_paths):
        try:
            features = {}
            # Call the main function with current args and extract to a dict
            features = extract_features_single_image(
                img_path,
                args.wavelet_type,
                args.level,
                args.lbp_P,
                args.lbp_R,
                args.lbp_method,
                args.hog_pixels_per_cell,
                args.hog_cells_per_block,
                args.n_segments,
                args.compactness,
                args.radius,
                args.feature_flags,
                args.output_dir
            )

            flat_feats = flatten_features(features)
            flat_feats['image'] = os.path.basename(img_path)
            all_data.append(flat_feats)

        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

    df = pd.DataFrame(all_data)

    # Move 'image' to the **first** column (you can swap to last if needed)
    cols = ['image'] + [col for col in df.columns if col != 'image']
    df = df[cols]
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    # Save to CSV
    df.to_csv(output_csv, index=False)

    print(f"\nSaved all features to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch extract features from all images in a directory.')
    parser.add_argument('input_dir', type=str, help='Directory containing input images')
    parser.add_argument('--output_csv', type=str, default='batch_features.csv', help='CSV file to save features')
    parser.add_argument('--output_dir', type=str, default='temp_batch', help='Intermediate output dir (if needed)')

    # Feature options (same as your main script)
    parser.add_argument('--wavelet_type', type=str, default='db1')
    parser.add_argument('--level', type=int, default=2)
    parser.add_argument('--lbp_P', type=int, default=8)
    parser.add_argument('--lbp_R', type=int, default=1)
    parser.add_argument('--lbp_method', type=str, default='uniform')
    parser.add_argument('--hog_pixels_per_cell', type=int, nargs=2, default=[8, 8])
    parser.add_argument('--hog_cells_per_block', type=int, nargs=2, default=[2, 2])
    parser.add_argument('--n_segments', type=int, default=100)
    parser.add_argument('--compactness', type=int, default=10)
    parser.add_argument('--radius', type=int, default=8)

    # Feature flags
    parser.add_argument('--glcm', action='store_true')
    parser.add_argument('--glrlm', action='store_true')
    parser.add_argument('--wavelet', action='store_true')
    parser.add_argument('--image_stats', action='store_true')
    parser.add_argument('--histogram', action='store_true')
    parser.add_argument('--gabor', action='store_true')
    parser.add_argument('--chip_histogram', action='store_true')
    parser.add_argument('--lbp', action='store_true')
    parser.add_argument('--hog', action='store_true')
    parser.add_argument('--fractal', action='store_true')
    parser.add_argument('--moments', action='store_true')
    parser.add_argument('--hu_moments', action='store_true')
    parser.add_argument('--zernike', action='store_true')
    parser.add_argument('--superpixel', action='store_true')

    args = parser.parse_args()

    # If no feature flags are given, set all to True
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
        'moments': args.moments,
        'hu_moments': args.hu_moments,
        'zernike': args.zernike,
        'superpixel': args.superpixel
    }

    if not any(feature_flags.values()):
        for k in feature_flags:
            feature_flags[k] = True

    args.feature_flags = feature_flags

    batch_extract(args.input_dir, args.output_csv, args)
