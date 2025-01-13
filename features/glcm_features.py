import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from scipy.stats import entropy as scipy_entropy
import math
def difference_distribution(glcm, num_levels= 64 ):
    """Calculate the difference distribution P_(x-y) from the GLCM."""
    P_diff = np.zeros(num_levels)
    for i in range(num_levels):
        for j in range(num_levels):
            diff = abs(i - j)
            P_diff[diff] += glcm[i, j]
    return P_diff
def calculate_sum_entropy(glcm, num_levels):
    """Calculate the sum entropy of the GLCM."""
    sum_entropy = 0
    for i in range(2, 2 * num_levels + 1):
        Pxpy = sum([glcm[j][k] for j in range(num_levels) for k in range(num_levels) if j + k + 2 == i])
        if Pxpy > 0:
            sum_entropy -= Pxpy * np.log(Pxpy)
    return sum_entropy

def calculate_sum_variance(glcm, num_levels):
    """Calculate the sum variance from the GLCM."""
    sum_entropy_value = calculate_sum_entropy(glcm, num_levels)
    sum_variance = 0
    for i in range(2, 2 * num_levels + 1):
        Pxpy = sum([glcm[j][k] for j in range(num_levels) for k in range(num_levels) if j + k + 2 == i])
        sum_variance += ((i - sum_entropy_value) ** 2) * Pxpy
    return sum_variance


def calculate_difference_variance(glcm):
    P_diff = difference_distribution(glcm)
    """Calculate the difference variance."""
    mu_diff = np.sum(np.arange(len(P_diff)) * P_diff)
    variance = np.sum((np.arange(len(P_diff)) - mu_diff) ** 2 * P_diff)
    return variance

def calculate_inverse_difference_moment(glcm):
    rows, cols = np.indices(glcm.shape[:2])
    idm = np.sum(glcm / (1 + (rows - cols) ** 2), axis=(0, 1))
    return idm

def calculate_sum_of_square_variances(glcm):
    # Compute the mean of the GLCM along the rows (mu_x)
    mu_x = np.sum(np.arange(glcm.shape[0]) * np.sum(glcm, axis=1))
    # Compute the SOSVH
    sosvh = 0
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[0]):
            sosvh += np.sum((i - mu_x)**2 * glcm[i, j])
    return sosvh

def calculate_inverse_difference_moment(glcm):
    """Calculate the Inverse Difference Moment (IDM) from a GLCM."""
    num_levels = glcm.shape[0]
    idm = 0
    
    for i in range(num_levels):
        for j in range(num_levels):
            idm += glcm[i, j] / (1 + (i - j) ** 2)
    
    return idm

def calculate_inverse_difference_moment_normalized(glcm):
    """Calculate the Normalized Inverse Difference Moment (IDM) from a GLCM."""
    idm = calculate_inverse_difference_moment(glcm)
    # Normalize IDM by dividing by the sum of the GLCM values
    glcm_sum = np.sum(glcm)
    normalized_idm = idm / glcm_sum
    
    return normalized_idm

def calculate_glcm_entropy(glcm):
    # Small value to avoid log(0)
    epsilon = 1e-10
    # Compute the entropy measure
    entropy = -np.sum(glcm * np.log(glcm + epsilon))
    return entropy

def calculate_renyi_entropy(glcm, alpha=2):
    # Flatten the GLCM and normalize it to get probabilities
    glcm_prob = glcm / np.sum(glcm)
    glcm_prob = glcm_prob[glcm_prob > 0]  # Filter out zero probabilities
    
    if alpha == 1:
        # Special case for alpha = 1, which is Shannon entropy
        return -np.sum(glcm_prob * np.log(glcm_prob))
    else:
        # General case for Rényi entropy
        return 1 / (1 - alpha) * np.log(np.sum(glcm_prob ** alpha))

def calculate_kapur_entropy(glcm, threshold = 128):
    # Normalize GLCM
    glcm_normalized = glcm / np.sum(glcm)
    
    # Compute cumulative sums for background and foreground probabilities
    cumsum = np.cumsum(glcm_normalized)
    # Probabilities for background and foreground
    p1 = cumsum[threshold]
    p2 = 1 - p1
    
    # Entropy for background and foreground regions
    if p1 == 0:
        entropy1 = 0
    else:
        entropy1 = -np.sum(glcm_normalized[:threshold] / p1 * np.log(glcm_normalized[:threshold] / p1 + 1e-10))
    
    if p2 == 0:
        entropy2 = 0
    else:
        entropy2 = -np.sum(glcm_normalized[threshold:] / p2 * np.log(glcm_normalized[threshold:] / p2 + 1e-10))
    
    return entropy1 + entropy2

def calculate_yager_entropy(glcm, w=0.5):
    # Normalize GLCM to get probabilities
    glcm_normalized = glcm / np.sum(glcm)
    
    # Filter out zero probabilities to avoid log(0) issues
    glcm_normalized = glcm_normalized[glcm_normalized > 0]
    
    # Compute Yager entropy
    entropy = -np.sum(glcm_normalized * (np.log(glcm_normalized) - np.log(1 - glcm_normalized)))
    return entropy

def calculate_sum_average(glcm):
    """Calculate the Sum Average from a GLCM."""
    num_levels = glcm.shape[0]
    sum_avg = 0
    
    # Calculate P_x+y(k) where k = i + j
    p_x_plus_y = np.zeros(2 * num_levels - 1)
    for i in range(num_levels):
        for j in range(num_levels):
            p_x_plus_y[i + j] += glcm[i, j]
    
    # Compute Sum Average
    for i in range(2, 2 * num_levels):
        sum_avg += i * p_x_plus_y[i - 2]  # Adjusting index by subtracting 2
    
    return sum_avg

def calculate_h_xy1(glcm):
    """Calculate H_{XY1} from a GLCM."""
    num_levels = glcm.shape[0]

    # Marginal probabilities
    p_x = np.sum(glcm, axis=1)  # Sum over columns to get row marginal probabilities
    p_y = np.sum(glcm, axis=0)  # Sum over rows to get column marginal probabilities

    h_xy1_value = 0

    # Calculate H_{XY1}
    for i in range(num_levels):
        for j in range(num_levels):
            if glcm[i, j] > 0:  # Only consider non-zero elements to avoid log(0)
                h_xy1_value -= glcm[i, j] * np.log(p_x[i] * p_y[j])

    return h_xy1_value

def calculate_HXY2(p_x, p_y):
    """Calculate H_{XY2}."""
    h_xy2 = 0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_x[i] > 0 and p_y[j] > 0:  # Avoid log(0)
                h_xy2 -= p_x[i] * p_y[j] * np.log(p_x[i] * p_y[j])
    return h_xy2

def calculate_information_measure_of_correlation1(glcm):
    hxy = calculate_glcm_entropy(glcm)
    hxy1 = calculate_h_xy1(glcm)
    imc1 = (hxy - hxy1) / np.maximum(hxy, hxy1)
    return imc1

def calculate_information_measure_of_correlation2(glcm):
    # Calculate joint entropy HXY
    hxy = calculate_glcm_entropy(glcm)
    
    # Marginal probabilities
    p_x = np.sum(glcm, axis=1)  # Sum over columns to get row marginal probabilities
    p_y = np.sum(glcm, axis=0)  # Sum over rows to get column marginal probabilities

    # Calculate H_{XY2}
    h_xy2 = calculate_HXY2(p_x, p_y)
    # Calculate IMC2
    imc2 = np.sqrt(1 - np.exp(-2 * (h_xy2 - hxy)))
    
    return imc2

def calculate_cluster_shade(glcm):
   # Compute the mean of the GLCM matrix along the x (rows) and y (columns) axes
    mu_x = np.mean(np.sum(glcm, axis=1))
    mu_y = np.mean(np.sum(glcm, axis=0))

# Compute the cs (cluster shade)
    Ng = glcm.shape[0]
    cs = 0
    for i in range(Ng):
        for j in range(Ng):
            cs += (i + j - mu_x - mu_y)**3 * glcm[i, j]
    return cs

def calculate_cluster_prominence(glcm):
   # Compute the mean of the GLCM matrix along the x (rows) and y (columns) axes
    mu_x = np.mean(np.sum(glcm, axis=1))
    mu_y = np.mean(np.sum(glcm, axis=0))

# Compute the cprom (cluster prominence)
    Ng = glcm.shape[0]
    cprom = 0
    for i in range(Ng):
        for j in range(Ng):
            cprom += (i + j - mu_x - mu_y)**4 * glcm[i, j]
    return cprom


def calculate_difference_entropy(glcm):
    P_diff = difference_distribution(glcm)
    """Calculate the difference entropy."""
    entropy = 0
    for p in P_diff:
        if p > 0:
            entropy -= p * np.log(p)
    return entropy

def rescale_image(image, levels=64):
    # Rescale the image to the specified number of levels
    image = (image / image.max()) * (levels - 1)
    return image.astype(np.uint8)

def extract_glcm_features(image, distances=[1], angles=[0], levels=64, symmetric=True, normed=True):
    # Rescale the image to the desired levels
    image = rescale_image(image, levels)
# Define the distances and angles for GLCM
    # distances = [1]  # You can change this to other distances as needed
    # angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles: 0, 45, 90, and 135 degrees
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normed)

    features = {}

    # Homogeneity 
    features['homogeneity '] = graycoprops(glcm, prop='homogeneity')

    # Contrast
    features['contrast '] = graycoprops(glcm, prop='contrast')
    
    # Energy 
    features['energy '] = graycoprops(glcm, prop='energy')

    # Dissimilarity
    features['dissimilarity '] = graycoprops(glcm, prop='dissimilarity')

    # Angular Second Moment (ASM)
    features['angular_second_moment'] = np.mean(graycoprops(glcm, 'ASM'))

    # Correlation
    features['correlation'] = np.mean(graycoprops(glcm, 'correlation'))

    features ['sum_average'] = calculate_sum_average(glcm)
    # Variance
    N = glcm.shape[0]
    # Calculate the mean (μ_x)
    mean_x = np.sum(np.arange(glcm.shape[0])[:, np.newaxis] * glcm)
    features ['mean_x'] = mean_x

    # Calculate the mean (μ_y)
    mean_y = np.sum(np.arange(glcm.shape[1]) * np.sum(glcm, axis=0))
    features ['mean_y'] = mean_y


    # Calculate the variance
    variance = 0.0
    for i in range(N):
        for j in range(N):
            variance += ((i - mean_x) ** 2) * glcm[i, j]
    features['variance'] = variance
    # standard deviation
    std_dv = math.sqrt(variance)
    features['standard_deviation'] = std_dv

    features ['maximal_prob'] = np.max(glcm)
    idm = calculate_inverse_difference_moment(glcm)
    features['idm'] = idm

    idm_normalized = calculate_inverse_difference_moment_normalized(glcm)
    features['idm_normalized'] = idm_normalized
    # Sum of Variance
    features['sum_of_variance'] = calculate_sum_variance(glcm, 64)

    # Difference of Variance
    features['difference_of_variance'] = calculate_difference_variance(glcm)

    # Inverse Difference Moment (IDM)
    # features['inverse_difference_moment'] =  calculate_inverse_difference_moment(glcm)
    # Sum of Squares: Variance
    features['sum_of_squares'] = calculate_sum_of_square_variances(glcm)

    # Entropy
    features['entropy'] = calculate_glcm_entropy(glcm)

    # Sum etnropy
    features['sum_entropy'] = calculate_sum_entropy(glcm, 64)

    # Difference etnropy
    features['difference_entropy'] = calculate_difference_entropy(glcm)

    # Renyi Entropy (with alpha = 2)
    features['renyi_entropy'] = calculate_renyi_entropy(glcm)

    # Yager Entropy (with w = 0.5)
    features['yager_entropy'] = calculate_yager_entropy(glcm)

    # Kapur Entropy
    features['kapur_entropy'] = calculate_kapur_entropy(glcm)

    # information_measure_of_correlation1
    features['information_measure_of_correlation1'] = calculate_information_measure_of_correlation1(glcm)

    # information_measure_of_correlation2
    features['information_measure_of_correlation2'] = calculate_information_measure_of_correlation2(glcm)

    # cluster_shade
    features['cluster_shade'] =  calculate_cluster_shade(glcm)

    # cluster_prominence
    features['cluster_prominence'] = calculate_cluster_prominence(glcm)
    # # Cluster Shade
    # ux = np.mean(np.arange(levels))
    # uy = np.mean(np.arange(levels))
    # cluster_shade = np.sum((np.arange(levels)[:, None] + np.arange(levels) - ux - uy) ** 3 * glcm)
    # features['cluster_shade'] = cluster_shade

    # # Cluster Prominence
    # cluster_prominence = np.sum((np.arange(levels)[:, None] + np.arange(levels) - ux - uy) ** 4 * glcm)
    # features['cluster_prominence'] = cluster_prominence

    return features
