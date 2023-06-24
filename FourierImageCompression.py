import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import cv2

def load_image(image_path):
    # Load the input image in grayscale format.
    return cv2.imread(image_path, 0)

def compute_fft(image):
    # Compute the Fast Fourier Transform of the input image.
    return fftpack.fftn(image)

def compress_image(image_fft, keep_fraction):
    # Compress the image by removing a percentage of frequency components.
    r, c = image_fft.shape
    image_fft[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
    image_fft[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
    return image_fft

def compute_metrics(image, image_new):
    # Compute the Mean Squared Error (MSE) between the original and compressed images.
    mse = np.mean((image - image_new) ** 2)
    # Computer the Peak Signal-to-Noise Ratio (PSNR) between the original and compressed images.
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return mse, psnr

def save_image(image_new, output_path):
    # Save the compressed image.
    cv2.imwrite(output_path, image_new)

def plot_images(image, image_new):
    # Plot the original and compressed images side by side.
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(1, 2, 2), plt.imshow(image_new, cmap='gray'), plt.title('Compressed Image')
    plt.show()