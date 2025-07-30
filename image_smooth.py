import numpy as np
import cv2

def gaussian_kernel(size: int, sigma: float):
    kernel_1d = np.linspace(-(size // 2), size // 2, size)
    kernel_1d = np.exp(-0.5 * (kernel_1d / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d

def gaussian_smoothing(image: np.ndarray, kernel: np.ndarray):
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2
    padded_image = np.pad(image, [(pad_h, pad_h), (pad_w, pad_w)], mode='reflect')

    smoothed_image = np.zeros_like(image, dtype=np.float64)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            smoothed_image[i, j] = np.sum(kernel * region)

    smoothed_image = np.clip(smoothed_image, 0, 255).astype(np.uint8)
    return smoothed_image

# Load and process image
image = cv2.imread('test_photos/test_gray.png', cv2.IMREAD_GRAYSCALE)
kernel_size = 5
sigma = 1.0
kernel = gaussian_kernel(kernel_size, sigma)
smoothed_image = gaussian_smoothing(image, kernel)

# Save the smoothed image
cv2.imwrite('test_photos/test_smoothed.png', smoothed_image)
