from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def grayscale_luminosity(image_path, output_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    if img_array.ndim == 3 and img_array.shape[2] == 3:
        gray_array = (0.21 * img_array[:, :, 0] +
                      0.72 * img_array[:, :, 1] +
                      0.07 * img_array[:, :, 2]).astype(np.uint8)
        
        gray_img = Image.fromarray(gray_array).convert('L')
        gray_img.save(output_path)
        
        plt.imshow(gray_array, cmap='gray')
        plt.axis('off')
    else:
        print("Error: Input image must be an RGB image.")

grayscale_luminosity(
    "test_photos/test.png",
    "test_photos/test_gray.png"
)