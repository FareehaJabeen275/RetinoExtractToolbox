import sys
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
from skimage import io
import numpy as np

# Add the path to the features directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'features')))

from features.glcm_features import extract_glcm_features
from features.wavelet_features import extract_wavelet_features
from features.image_statistics import extract_image_statistics
from features.histogram_based_statistics import extract_histogram_features
from features.gabor_features import extract_gabor_features
from features.chip_histogram_features import extract_chip_histogram_features
from features.LBP_features import extract_lbp_features
from features.HOG_features import extract_hog_features
from features.fractal_dimension import extract_fractal_dimension
from features.superpixel_processing import extract_superpixels, compute_superpixel_features, visualize_superpixels
from features.moments_all import compute_moments
from input.image_loader import preprocess_image, load_image

def extract_features(image_path, wavelet, level, lbp_P, lbp_R, lbp_method, hog_pixels_per_cell, hog_cells_per_block, n_segments, compactness, save_csv, extract_glcm, extract_wavelet, extract_statistics, extract_histogram, extract_gabor, extract_chip, extract_lbp, extract_hog, extract_fractal, extract_superpixel, extract_moments):
    # Load and preprocess the image
    image = load_image(image_path, color_mode='grayscale')
    preprocessed_image = preprocess_image(image)
    image1 = load_image(image_path, color_mode='color')
    original_image, segments, segmented_image = extract_superpixels(image1, n_segments=n_segments, compactness=compactness)
        
    # Convert the preprocessed image back to uint8 format
    preprocessed_image_uint8 = (preprocessed_image * 255).astype(np.uint8)

    features = {}

    # Extract features based on checkboxes
    if extract_glcm:
        features['GLCM'] = extract_glcm_features(preprocessed_image_uint8)
    if extract_wavelet:
        features['Wavelet'] = extract_wavelet_features(preprocessed_image_uint8, wavelet=wavelet, level=level)
    if extract_statistics:
        features['Image Statistics'] = extract_image_statistics(preprocessed_image_uint8)
    if extract_histogram:
        features['Histogram Based Statistics'] = extract_histogram_features(preprocessed_image_uint8)
    if extract_gabor:
        features['Gabor Features'] = extract_gabor_features(preprocessed_image_uint8)
    if extract_chip:
        chip_coords = (50, 50, 100, 100)  # Example coordinates
        features['Chip Histogram Features'] = extract_chip_histogram_features(image, chip_coords)
    if extract_lbp:
        features['LBP Features'] = extract_lbp_features(preprocessed_image_uint8, P=lbp_P, R=lbp_R, method=lbp_method)
    if extract_hog:
        features['HOG Features'], _ = extract_hog_features(preprocessed_image_uint8, pixels_per_cell=tuple(hog_pixels_per_cell), cells_per_block=tuple(hog_cells_per_block), visualize=True)
    if extract_fractal:
        features['Fractal Dimension'] = extract_fractal_dimension(preprocessed_image_uint8)
    if extract_superpixel:
        features['Superpixel Features'] = compute_superpixel_features(image1, segments)
        visualize_superpixels(original_image, segments, segmented_image)
    if extract_moments:
        features['Moment Features'] = compute_moments(preprocessed_image_uint8)

    # Prepare features for display
    features_text = '\n\n'.join([f"{key}: {value}" for key, value in features.items()])
    
    # Save features to CSV
    if save_csv:
        features_df = pd.DataFrame({key: [value] for key, value in features.items()})
        csv_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if csv_path:
            features_df.to_csv(csv_path, index=False)
            messagebox.showinfo("Success", "Features saved successfully!")

    # Update the text widget with the features
    features_text_widget.config(state=tk.NORMAL)
    features_text_widget.delete(1.0, tk.END)
    features_text_widget.insert(tk.END, features_text)
    features_text_widget.config(state=tk.DISABLED)
    messagebox.showinfo("Info", "Feature extraction completed.")

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        image_path_label.config(text=file_path)

def run_extraction():
    image_path = image_path_label.cget("text")
    if not image_path:
        messagebox.showwarning("Warning", "Please select an image file.")
        return
    
    # Convert GUI inputs to tuples
    try:
        hog_pixels_per_cell = tuple(map(int, hog_pixels_per_cell_var.get().split(',')))
        hog_cells_per_block = tuple(map(int, hog_cells_per_block_var.get().split(',')))
    except ValueError:
        messagebox.showerror("Error", "Invalid format for HOG parameters. Please use 'x,y' format.")
        return

    save_csv = save_csv_var.get()

    extract_features(image_path, wavelet_var.get(), level_var.get(), lbp_P_var.get(), lbp_R_var.get(), lbp_method_var.get(), 
                     hog_pixels_per_cell, hog_cells_per_block, n_segments_var.get(), compactness_var.get(), save_csv,
                     extract_glcm_var.get(), extract_wavelet_var.get(), extract_statistics_var.get(), extract_histogram_var.get(),
                     extract_gabor_var.get(), extract_chip_var.get(), extract_lbp_var.get(), extract_hog_var.get(), 
                     extract_fractal_var.get(), extract_superpixel_var.get(), extract_moments_var.get())

# Set up the GUI
root = tk.Tk()
root.title("RetinoExtract Toolbox")
root.geometry("1100x900")
# root.configure(bg='#d8bfd8')
root.configure(bg='#d8bfd8')

# Create a canvas and scrollbar
canvas = tk.Canvas(root, bg='#d8bfd8')
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg='#d8bfd8')

# Configure scrollbar and canvas
scrollbar.grid(row=0, column=1, sticky='ns')
canvas.grid(row=0, column=0, sticky='nsew')
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

# Set grid weight to ensure resizing
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Create a grid layout in the scrollable frame
for i in range(0, 30):
    scrollable_frame.grid_rowconfigure(i, weight=1)
for i in range(0, 3):
    scrollable_frame.grid_columnconfigure(i, weight=1)

# Add widgets to the scrollable frame
tk.Label(scrollable_frame, text="Image File:", bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=0, column=0, padx=10, pady=10, sticky='w')
image_path_label = tk.Label(scrollable_frame, text="", width=100, anchor='w', bg='#ffffff', relief='sunken', font=('Arial', 10))
image_path_label.grid(row=0, column=1, padx=10, pady=10, sticky='w')

tk.Button(scrollable_frame, text="Open File", command=open_file, bg='#00796b', fg='#ffffff', font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=10, pady=10)

# Define variables for checkbuttons
extract_glcm_var = tk.BooleanVar()
extract_wavelet_var = tk.BooleanVar()
extract_statistics_var = tk.BooleanVar()
extract_histogram_var = tk.BooleanVar()
extract_gabor_var = tk.BooleanVar()
extract_chip_var = tk.BooleanVar()
extract_lbp_var = tk.BooleanVar()
extract_hog_var = tk.BooleanVar()
extract_fractal_var = tk.BooleanVar()
extract_superpixel_var = tk.BooleanVar()
extract_moments_var = tk.BooleanVar()

# Two-column layout for options
options_frame = tk.Frame(scrollable_frame, bg='#d8bfd8')
options_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky='ew')

# Column 1
tk.Label(options_frame, text="Wavelet:", bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=0, column=0, padx=10, pady=10, sticky='w')
wavelet_var = tk.StringVar(value='db1')
tk.Entry(options_frame, textvariable=wavelet_var, font=('Arial', 10)).grid(row=0, column=1, padx=10, pady=10)

tk.Label(options_frame, text="Level:", bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=1, column=0, padx=10, pady=10, sticky='w')
level_var = tk.IntVar(value=1)
tk.Entry(options_frame, textvariable=level_var, font=('Arial', 10)).grid(row=1, column=1, padx=10, pady=10)

tk.Label(options_frame, text="LBP P:", bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=2, column=0, padx=10, pady=10, sticky='w')
lbp_P_var = tk.IntVar(value=8)
tk.Entry(options_frame, textvariable=lbp_P_var, font=('Arial', 10)).grid(row=2, column=1, padx=10, pady=10)

tk.Label(options_frame, text="LBP R:", bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=3, column=0, padx=10, pady=10, sticky='w')
lbp_R_var = tk.IntVar(value=1)
tk.Entry(options_frame, textvariable=lbp_R_var, font=('Arial', 10)).grid(row=3, column=1, padx=10, pady=10)

tk.Label(options_frame, text="LBP Method:", bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=4, column=0, padx=10, pady=10, sticky='w')
lbp_method_var = tk.StringVar(value='uniform')
tk.Entry(options_frame, textvariable=lbp_method_var, font=('Arial', 10)).grid(row=4, column=1, padx=10, pady=10)

tk.Label(options_frame, text="HOG Pixels Per Cell (x,y):", bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=5, column=0, padx=10, pady=10, sticky='w')
hog_pixels_per_cell_var = tk.StringVar(value='8,8')
tk.Entry(options_frame, textvariable=hog_pixels_per_cell_var, font=('Arial', 10)).grid(row=5, column=1, padx=10, pady=10)

tk.Label(options_frame, text="HOG Cells Per Block (x,y):", bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=6, column=0, padx=10, pady=10, sticky='w')
hog_cells_per_block_var = tk.StringVar(value='3,3')
tk.Entry(options_frame, textvariable=hog_cells_per_block_var, font=('Arial', 10)).grid(row=6, column=1, padx=10, pady=10)

tk.Label(options_frame, text="Number of Segments:", bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=7, column=0, padx=10, pady=10, sticky='w')
n_segments_var = tk.IntVar(value=100)
tk.Entry(options_frame, textvariable=n_segments_var, font=('Arial', 10)).grid(row=7, column=1, padx=10, pady=10)

tk.Label(options_frame, text="Compactness:", bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=8, column=0, padx=10, pady=10, sticky='w')
compactness_var = tk.DoubleVar(value=0.1)
tk.Entry(options_frame, textvariable=compactness_var, font=('Arial', 10)).grid(row=8, column=1, padx=10, pady=10)

tk.Label(options_frame, text="Save CSV:", bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=9, column=0, padx=10, pady=10, sticky='w')
save_csv_var = tk.BooleanVar()
tk.Checkbutton(options_frame, variable=save_csv_var, bg='#d8bfd8').grid(row=9, column=1, padx=10, pady=10, sticky='w')

# Column 2
tk.Checkbutton(options_frame, text="Extract GLCM Features", variable=extract_glcm_var, bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=0, column=2, padx=10, pady=5, sticky='w')
tk.Checkbutton(options_frame, text="Extract Wavelet Features", variable=extract_wavelet_var, bg='#d8bfd8',font=('Arial', 12, 'bold')).grid(row=1, column=2, padx=10, pady=5, sticky='w')
tk.Checkbutton(options_frame, text="Extract Image Statistics", variable=extract_statistics_var, bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=2, column=2, padx=10, pady=5, sticky='w')
tk.Checkbutton(options_frame, text="Extract Histogram Based Statistics", variable=extract_histogram_var, bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=3, column=2, padx=10, pady=5, sticky='w')
tk.Checkbutton(options_frame, text="Extract Gabor Features", variable=extract_gabor_var, bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=4, column=2, padx=10, pady=5, sticky='w')
tk.Checkbutton(options_frame, text="Extract Chip Histogram Features", variable=extract_chip_var, bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=5, column=2, padx=10, pady=5, sticky='w')
tk.Checkbutton(options_frame, text="Extract LBP Features", variable=extract_lbp_var, bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=6, column=2, padx=10, pady=5, sticky='w')
tk.Checkbutton(options_frame, text="Extract HOG Features", variable=extract_hog_var, bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=7, column=2, padx=10, pady=5, sticky='w')
tk.Checkbutton(options_frame, text="Extract Fractal Dimension", variable=extract_fractal_var, bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=8, column=2, padx=10, pady=5, sticky='w')
tk.Checkbutton(options_frame, text="Extract Superpixel Features", variable=extract_superpixel_var, bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=9, column=2, padx=10, pady=5, sticky='w')
tk.Checkbutton(options_frame, text="Extract Moment Features", variable=extract_moments_var, bg='#d8bfd8', font=('Arial', 12, 'bold')).grid(row=10, column=2, padx=10, pady=5, sticky='w')

# Add a text widget for displaying features
features_text_widget = tk.Text(scrollable_frame, wrap='word', height=15, width=100, bg='#f1f8e9', font=('Arial', 10))
features_text_widget.grid(row=2, column=0, columnspan=3, padx=10, pady=10)
features_text_widget.config(state=tk.DISABLED)

# Add a button to run feature extraction
tk.Button(scrollable_frame, text="Run Extraction", command=run_extraction, bg='#4CAF50', fg='#ffffff', font=('Arial', 12, 'bold')).grid(row=3, column=0, columnspan=3, padx=10, pady=20)

root.mainloop()
