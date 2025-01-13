import cv2
import numpy as np

def compute_moments(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Calculate moments
    moments = cv2.moments(gray_image)

    # Hu Moments (invariant moments)
    hu_moments = cv2.HuMoments(moments).flatten()

    return {
        'm00': moments['m00'],
        'm10': moments['m10'],
        'm01': moments['m01'],
        'm20': moments['m20'],
        'm11': moments['m11'],
        'm02': moments['m02'],
        'm30': moments['m30'],
        'm21': moments['m21'],
        'm12': moments['m12'],
        'm03': moments['m03'],
        'mu20': moments['mu20'],
        'mu11': moments['mu11'],
        'mu02': moments['mu02'],
        'mu30': moments['mu30'],
        'mu21': moments['mu21'],
        'mu12': moments['mu12'],
        'mu03': moments['mu03'],
        'nu20': moments['nu20'],
        'nu11': moments['nu11'],
        'nu02': moments['nu02'],
        'nu30': moments['nu30'],
        'nu21': moments['nu21'],
        'nu12': moments['nu12'],
        'nu03': moments['nu03']
    }
