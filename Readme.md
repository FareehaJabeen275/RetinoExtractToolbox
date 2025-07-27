**RetinoExtract Toolbox**

**Table of Contents**

- [Overview](#overview)
- [Key Features](#key-features)
- [Supported Feature Extractors](#supported-feature-extractors)
- [Installation](#installation)
- [Usage](#usage)
  - [GUI](#graphical-user-interface-gui)
  - [CLI](#command-line-interface-cli)
  - [Available Options ](#Available Options )
- [Dependency Rules](#dependency-rules)
- [Valid Option Values](#valid-option-values)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [Example Data](#example-data)
- [Reference](#reference)
- [Contributing](#contributing)
- [License](#license)

### **Overview**
RetinoExtract is an open-source Python toolbox for automated feature extraction from retinal fundus images. It supports both a Graphical User Interface (GUI) and Command Line Interface (CLI), making it accessible for users with varying technical expertise.

### **Key Features:**
1. GUI support: Load images, select feature types, configure parameters, and extract features with a few clicks. Results can be saved as CSV.
2. CLI interface: Automate batch feature extraction with full control over options and parameters.

### **Supported Feature Extractors**
The toolbox supports the following:
1. Feature Extraction:
    Texture (GLCM, GLRLM, LBP)
    Intensity and Histogram
    Wavelet Decomposition
    Shape (Hu Moments, Zernike Moments)
    Edge Detection, Histogram of Oriented Gradients (HOG)
    Fractal Dimensions
    Gabor Filters
    Superpixel Analysis
2. Customizable Parameters for each feature type.
3. Save Results: Export extracted feature values as CSV files.

Make sure to include main options when using dependent parameters. Otherwise, the toolbox will raise an error. 
For Example: Ensure you provide main feature flags (e.g., --wavelet) when using dependent parameters (e.g., --wavelet_type), otherwise the toolbox will raise an error.

 ### **Installation**
1.	Clone the repository
‘git clone https://github.com/yourusername/RetinoExtractToolbox.git’
2.	Navigate to the directory
‘cd RetinoExtractToolbox’
3.	Install required dependencies
‘pip install -r requirements.txt’
### **Usage**
### **Graphical User Interface (GUI)**

```bash
python gui.py
```

Steps:
•	**Load Image**: Use the "Load Image" button to select a fundus image.
•	**Select Features**: Choose feature types (e.g., GLCM, LBP).
•	**Set Parameters**: Configure feature-specific parameters.
•	**Extract Features**: Click "Extract Features" to display results.
•	**Save CSV**: Optional checkbox to save results.

**Screenshot**
![GUI Screenshot](gui_screenshot.png)

### **Command Line Interface (CLI)**
```bash
python main.py [OPTIONS] image_path
```

**Example Command:**
```bash
python main.py glcm lbp_P 8 lbp_R 1 --output_dir results/ path/to/image.jpg
```

### **Available Options:**
| Option                  | Description                                    | Example                     |
| ----------------------- | ---------------------------------------------- | --------------------------- |
| `--glcm`                | Extract GLCM (Gray Level Co-occurrence Matrix) | `--glcm`                    |
| `--glrlm`               | Extract GLRLM (Gray Level Run Length Matrix)   | `--glrlm`                   |
| `--wavelet`             | Extract wavelet features                       | `--wavelet`                 |
| `--wavelet_type`        | Wavelet type                                   | `--wavelet_type db1`        |
| `--level`               | Decomposition level                            | `--level 3`                 |
| `--lbp`                 | Local Binary Pattern features                  | `--lbp`                     |
| `--lbp_P`               | LBP points                                     | `--lbp_P 8`                 |
| `--lbp_R`               | LBP radius                                     | `--lbp_R 1`                 |
| `--lbp_method`          | LBP method                                     | `--lbp_method uniform`      |
| `--hog`                 | Histogram of Oriented Gradients                | `--hog`                     |
| `--hog_pixels_per_cell` | Pixels per cell (HOG)                          | `--hog_pixels_per_cell 8 8` |
| `--hog_cells_per_block` | Cells per block (HOG)                          | `--hog_cells_per_block 2 2` |
| `--superpixel`          | Superpixel-based features                      | `--superpixel`              |
| `--n_segments`          | Number of superpixels                          | `--n_segments 100`          |
| `--compactness`         | Compactness parameter                          | `--compactness 10`          |
| `--radius`              | Radius for circular features                   | `--radius 5`                |
| `--output_dir`          | Directory to save results                      | `--output_dir results/`     |
| `--image_stats`         | Basic image statistics                         | `--image_stats`             |
| `--histogram`           | Histogram-based features                       | `--histogram`               |
| `--gabor`               | Gabor filter-based features                    | `--gabor`                   |
| `--chip_histogram`      | Histogram features for image chips             | `--chip_histogram`          |
| `--fractal`             | Fractal dimension features                     | `--fractal`                 |
| `--moments`             | Statistical moments                            | `--moments`                 |
| `--hu_moments`          | Hu shape moments                               | `--hu_moments`              |
| `--zernike`             | Zernike shape moments                          | `--zernike`                 |

 ### **Dependency Rules**
Use the following together to avoid errors:
--wavelet_type     requires --wavelet
--level            requires --wavelet
--lbp_P            requires --lbp
--lbp_R            requires --lbp
--lbp_method       requires --lbp
--hog_pixels_per_cell  requires --hog
--hog_cells_per_block  requires --hog
--n_segments       requires --superpixel
--compactness      requires --superpixel
--radius           requires --superpixel

### **Valid Option Values**
•	`--wavelet_type`: haar, db1, sym2, coif1, etc.
•	`--lbp_method`: uniform, default, ror, var
•	`--hog_pixels_per_cell`: e.g., 8 8
•	`--hog_cells_per_block`: e.g., 2 2
•	`--compactness`, `--n_segments`: Positive integers

### **Error Handling**
If required arguments are missing:
```text
usage: main.py [h] [--wavelet_type WAVELET_TYPE] [--level LEVEL] ...
main.py: error: the following arguments are required: image_path
```
### **Examples**
1. **Extract all features (CLI):**
```bash
python main.py glcm lbp wavelet --output_dir output/ image.jpg
```
2. **Save results as CSV (GUI):**
   Check the "Save CSV" option → Click "Extract Features".
### **Example Data**
You may use publicly available datasets:
1.	DRIVE
2.	ORIGA
3.	REFUGE
4.	DRISHTI-GS
5.	RIM-ONE

### **Reference**
If you use this toolbox for research or projects, please cite:
Jabeen et al. (2025). "RetinoExtract Toolbox: A Feature Extraction Tool for Fundus Images."

### **Contributing**
See `CONTRIBUTING.md` for contribution guidelines.
 
### **License**
Distributed under the terms of the MIT license.
See [LICENSE](LICENSE) for details.

