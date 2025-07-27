

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from scipy.stats import entropy as scipy_entropy
import math

def difference_distribution(glcm, num_levels=64):
    P_diff = np.zeros(num_levels)
    for i in range(num_levels):
        for j in range(num_levels):
            diff = abs(i - j)
            P_diff[diff] += glcm[i, j, 0, 0].item()
    return P_diff

def calculate_sum_entropy(glcm, num_levels):
    sum_entropy = 0
    for i in range(2, 2 * num_levels + 1):
        Pxpy = sum([glcm[j, k, 0, 0].item() for j in range(num_levels) for k in range(num_levels) if j + k + 2 == i])
        if Pxpy > 0:
            sum_entropy -= Pxpy * np.log(Pxpy)
    return sum_entropy

def calculate_sum_variance(glcm, num_levels):
    sum_entropy_value = calculate_sum_entropy(glcm, num_levels)
    sum_variance = 0
    for i in range(2, 2 * num_levels + 1):
        Pxpy = sum([glcm[j, k, 0, 0].item() for j in range(num_levels) for k in range(num_levels) if j + k + 2 == i])
        sum_variance += ((i - sum_entropy_value) ** 2) * Pxpy
    return sum_variance

def calculate_difference_variance(glcm):
    P_diff = difference_distribution(glcm)
    mu_diff = np.sum(np.arange(len(P_diff)) * P_diff)
    variance = np.sum((np.arange(len(P_diff)) - mu_diff) ** 2 * P_diff)
    return variance

def calculate_inverse_difference_moment(glcm):
    num_levels = glcm.shape[0]
    idm = 0
    for i in range(num_levels):
        for j in range(num_levels):
            idm += glcm[i, j, 0, 0].item() / (1 + (i - j) ** 2)
    return idm

def calculate_inverse_difference_moment_normalized(glcm):
    idm = calculate_inverse_difference_moment(glcm)
    glcm_sum = np.sum(glcm)
    normalized_idm = idm / glcm_sum
    return normalized_idm

def calculate_glcm_entropy(glcm):
    epsilon = 1e-10
    entropy = -np.sum(glcm * np.log(glcm + epsilon))
    return entropy

def calculate_renyi_entropy(glcm, alpha=2):
    glcm_prob = glcm / np.sum(glcm)
    glcm_prob = glcm_prob[glcm_prob > 0]
    if alpha == 1:
        return -np.sum(glcm_prob * np.log(glcm_prob))
    else:
        return 1 / (1 - alpha) * np.log(np.sum(glcm_prob ** alpha))

def calculate_kapur_entropy(glcm, threshold=128):
    glcm_normalized = glcm / np.sum(glcm)
    cumsum = np.cumsum(glcm_normalized)
    p1 = cumsum[threshold]
    p2 = 1 - p1
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
    glcm_normalized = glcm / np.sum(glcm)
    glcm_normalized = glcm_normalized[glcm_normalized > 0]
    entropy = -np.sum(glcm_normalized * (np.log(glcm_normalized) - np.log(1 - glcm_normalized)))
    return entropy

def calculate_sum_average(glcm):
    num_levels = glcm.shape[0]
    sum_avg = 0
    p_x_plus_y = np.zeros(2 * num_levels - 1)
    for i in range(num_levels):
        for j in range(num_levels):
            p_x_plus_y[i + j] += glcm[i, j, 0, 0].item()
    for i in range(2, 2 * num_levels):
        sum_avg += i * p_x_plus_y[i - 2]
    return sum_avg

def calculate_h_xy1(glcm):
    num_levels = glcm.shape[0]
    p_x = np.sum(glcm, axis=1)  # Shape: (num_levels, 1, 1)
    p_y = np.sum(glcm, axis=0)  # Shape: (num_levels, 1, 1)
    h_xy1_value = 0
    for i in range(num_levels):
        for j in range(num_levels):
            pij = glcm[i, j, 0, 0].item()
            if pij > 0:
                px = p_x[i, 0, 0].item()
                py = p_y[j, 0, 0].item()
                if px * py > 0:
                    h_xy1_value -= pij * np.log(px * py)
    return h_xy1_value


def calculate_HXY2(p_x, p_y):
    h_xy2 = 0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_x[i] > 0 and p_y[j] > 0:
                h_xy2 -= p_x[i] * p_y[j] * np.log(p_x[i] * p_y[j])
    return h_xy2

def calculate_information_measure_of_correlation1(glcm):
    hxy = calculate_glcm_entropy(glcm)
    hxy1 = calculate_h_xy1(glcm)
    imc1 = (hxy - hxy1) / np.maximum(hxy, hxy1)
    return imc1

def calculate_information_measure_of_correlation2(glcm):
    hxy = calculate_glcm_entropy(glcm)
    p_x = np.sum(glcm, axis=1)[:, 0, 0]
    p_y = np.sum(glcm, axis=0)[0, :, 0]
    h_xy2 = calculate_HXY2(p_x, p_y)
    exponent = -2 * (h_xy2 - hxy)
    exp_val = np.exp(exponent)
    value_inside_sqrt = max(0.0, 1 - exp_val)
    imc2 = np.sqrt(value_inside_sqrt)
    return imc2

def calculate_cluster_shade(glcm):
    mu_x = np.mean(np.sum(glcm, axis=1))
    mu_y = np.mean(np.sum(glcm, axis=0))
    Ng = glcm.shape[0]
    cs = 0
    for i in range(Ng):
        for j in range(Ng):
            cs += (i + j - mu_x - mu_y) ** 3 * glcm[i, j, 0, 0].item()
    return cs

def calculate_cluster_prominence(glcm):
    mu_x = np.mean(np.sum(glcm, axis=1))
    mu_y = np.mean(np.sum(glcm, axis=0))
    Ng = glcm.shape[0]
    cprom = 0
    for i in range(Ng):
        for j in range(Ng):
            cprom += (i + j - mu_x - mu_y) ** 4 * glcm[i, j, 0, 0].item()
    return cprom

def calculate_difference_entropy(glcm):
    P_diff = difference_distribution(glcm)
    entropy = 0
    for p in P_diff:
        if p > 0:
            entropy -= p * np.log(p)
    return entropy

def rescale_image(image, levels=64):
    image = (image / image.max()) * (levels - 1)
    return image.astype(np.uint8)

def extract_glcm_features(image, distances=[1], angles=[0], levels=64, symmetric=True, normed=True):
    image = rescale_image(image, levels)
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normed)
    features = {}
    features['homogeneity '] = graycoprops(glcm, prop='homogeneity')
    features['contrast '] = graycoprops(glcm, prop='contrast')
    features['energy '] = graycoprops(glcm, prop='energy')
    features['dissimilarity '] = graycoprops(glcm, prop='dissimilarity')
    features['angular_second_moment'] = np.mean(graycoprops(glcm, 'ASM'))
    features['correlation'] = np.mean(graycoprops(glcm, 'correlation'))
    features['sum_average'] = calculate_sum_average(glcm)
    N = glcm.shape[0]
    mean_x = np.sum(np.arange(glcm.shape[0])[:, np.newaxis] * np.sum(glcm, axis=1)[:, 0, 0])
    features['mean_x'] = mean_x
    mean_y = np.sum(np.arange(glcm.shape[1]) * np.sum(glcm, axis=0)[0, :, 0])
    features['mean_y'] = mean_y
    variance = 0.0
    for i in range(N):
        for j in range(N):
            variance += ((i - mean_x) ** 2) * glcm[i, j, 0, 0].item()
    features['variance'] = variance
    std_dv = math.sqrt(variance)
    features['standard_deviation'] = std_dv
    features['maximal_prob'] = np.max(glcm)
    features['idm'] = calculate_inverse_difference_moment(glcm)
    features['idm_normalized'] = calculate_inverse_difference_moment_normalized(glcm)
    features['sum_of_variance'] = calculate_sum_variance(glcm, 64)
    features['difference_of_variance'] = calculate_difference_variance(glcm)
    features['sum_of_squares'] = np.sum([(i - mean_x) ** 2 * np.sum(glcm[i, :, 0, 0]) for i in range(N)])
    features['entropy'] = calculate_glcm_entropy(glcm)
    features['sum_entropy'] = calculate_sum_entropy(glcm, 64)
    features['difference_entropy'] = calculate_difference_entropy(glcm)
    features['renyi_entropy'] = calculate_renyi_entropy(glcm)
    features['yager_entropy'] = calculate_yager_entropy(glcm)
    features['kapur_entropy'] = calculate_kapur_entropy(glcm)
    features['information_measure_of_correlation1'] = calculate_information_measure_of_correlation1(glcm)
    features['information_measure_of_correlation2'] = calculate_information_measure_of_correlation2(glcm)
    features['cluster_shade'] = calculate_cluster_shade(glcm)
    features['cluster_prominence'] = calculate_cluster_prominence(glcm)
    return features
