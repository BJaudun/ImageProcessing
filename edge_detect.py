import numpy as np
import cv2
import os

def gaussian_kernel(size: int, sigma: float):
    kernel_1d = np.linspace(-(size // 2), size // 2, size)
    kernel_1d = np.exp(-0.5 * (kernel_1d / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d

def gaussian_smoothing(image: np.ndarray, kernel: np.ndarray):
    kernel_height, kernel_width = kernel.shape
    padded_image = np.pad(image, [(kernel_height // 2, kernel_height // 2),
                                  (kernel_width // 2, kernel_width // 2)], mode='constant', constant_values=0)
    smoothed_image = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            smoothed_image[i, j] = np.sum(kernel * region)
    return smoothed_image

def sobel_operator(image: np.ndarray):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    grad_x = cv2.filter2D(image, cv2.CV_32F, sobel_x)
    grad_y = cv2.filter2D(image, cv2.CV_32F, sobel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)

image = cv2.imread('test_photos/test_gray.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found at 'test_photos/test_gray.png'.")

kernel_size = 5
sigma = 1.0
kernel = gaussian_kernel(kernel_size, sigma)
smoothed_image = gaussian_smoothing(image, kernel)
edges = sobel_operator(smoothed_image)

output_path = 'test_photos/edges_output.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
cv2.imwrite(output_path, edges)
