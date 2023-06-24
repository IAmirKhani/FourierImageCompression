import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import cv2
import os

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

if __name__ == "__main__":
    image_path = 'input.jpg'
    output_path = 'output_image.jpg'
    keep_fraction = 0.1

    image = load_image(image_path)
    image_fft = compute_fft(image)
    
    # Visualize the frequency spectrum of the original image
    plt.subplot(2,2,1), plt.imshow(np.log(1+np.abs(image_fft)), cmap='gray')
    plt.title('Original Image FFT')
    
    image_fft_compressed = compress_image(image_fft, keep_fraction)
    
    # Visualize the frequency spectrum of the compressed image
    plt.subplot(2,2,2), plt.imshow(np.log(1+np.abs(image_fft_compressed)), cmap='gray')
    plt.title(f'Compressed Image FFT ({keep_fraction*100}% frequency components)')
    
    image_new = fftpack.ifftn(image_fft_compressed).real

    mse, psnr = compute_metrics(image, image_new)
    print(f"MSE: {mse}")
    print(f"PSNR: {psnr}")

    save_image(image_new, output_path)

    # Analyze image file sizes
    orig_size = os.stat('input.jpg').st_size
    compressed_size = os.stat("output_image.jpg").st_size
    print(f"Original image size: {orig_size} bytes")
    print(f"Compressed image size: {compressed_size} bytes")
    print(f"Size difference: {orig_size - compressed_size} bytes")
    
    # Visualize the original image and visualize the compressed image.
    plt.subplot(2,2,3), plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(2,2,4), plt.imshow(image_new, cmap='gray')
    plt.title('Compressed Image')
    plt.tight_layout()
    plt.show()