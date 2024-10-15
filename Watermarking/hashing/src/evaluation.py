from math import log10
import numpy as np
import cv2 as cv


def ssim_similarity(image_x: np.ndarray, image_y: np.ndarray, k1: float = 0.01, k2: float = 0.03) -> float:
    assert image_x.shape == image_y.shape, "shapes of images mismatch"
    assert image_x.dtype == np.uint8 and image_y.dtype == np.uint8, "both images must be of 8-bit type"
    c1 = (k1 * 255) ** 2
    c2 = (k2 * 255) ** 2
    n = image_x.size
    x_mean = float(np.sum(image_x, dtype=np.int64)) / n
    y_mean = float(np.sum(image_y, dtype=np.int64)) / n
    x_minus_mean = image_x.astype(np.float64) - np.full_like(image_x, x_mean, dtype=np.float64)
    y_minus_mean = image_y.astype(np.float64) - np.full_like(image_y, y_mean, dtype=np.float64)
    x_variance = float(np.sum(np.square(x_minus_mean))) / (n - 1)
    y_variance = float(np.sum(np.square(y_minus_mean))) / (n - 1)
    covariance = float(np.sum(np.multiply(x_minus_mean, y_minus_mean))) / (n - 1)
    return (((2 * x_mean * y_mean + c1) * (2 * covariance + c2)) 
            / ((x_mean ** 2 + y_mean ** 2 + c1) * (x_variance + y_variance + c2)))


def mse(image_x: np.ndarray, image_y: np.ndarray) -> float:
    assert image_x.shape == image_y.shape, "shapes of images mismatch"
    assert image_x.dtype == np.uint8 and image_y.dtype == np.uint8, "both images must be of 8-bit type"
    return float(np.mean(np.square(cv.absdiff(image_x, image_y), dtype=np.uint16)))


def psnr(image_x: np.ndarray, image_y: np.ndarray) -> tuple[float, float]:
    """Returns (PSNR, MSE) for two images."""
    _mse = mse(image_x, image_y)
    return 10 * log10(255 ** 2 / _mse), _mse
