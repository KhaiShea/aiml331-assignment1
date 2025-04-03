import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output folder exists
os.makedirs("images", exist_ok=True)

# === Load grayscale checkerboard image ===
img = cv2.imread("images/checkerboard.png", cv2.IMREAD_GRAYSCALE)

# === 2.2 Histogram and CDF ===
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
cdf = hist.cumsum() / hist.sum()

plt.figure()
plt.plot(cdf)
plt.title("CDF of Original Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Cumulative Probability")
plt.savefig("images/cdf.png")
plt.close()

# === 2.3 Gaussian Blur (Spatial Low-pass Filter) ===
blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
cv2.imwrite("images/blurred.png", blurred)

# === 2.5 FFT-based Frequency Domain Low-pass Filter ===
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# Create a low-pass filter mask
mask = np.zeros_like(img)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

filtered_fshift = fshift * mask
f_ishift = np.fft.ifftshift(filtered_fshift)
filtered_img = np.fft.ifft2(f_ishift).real.astype(np.uint8)

cv2.imwrite("images/frequency_filtered.png", filtered_img)

# === 2.6 Histogram of Frequency Filtered Image ===
filtered_hist = cv2.calcHist([filtered_img], [0], None, [256], [0, 256])
plt.figure()
plt.plot(filtered_hist)
plt.title("Histogram of Frequency Filtered Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.savefig("images/frequency_histogram.png")
plt.close()

# === 2.7 Histogram Equalisation using CDF of Blurred Image ===
blurred_hist = cv2.calcHist([blurred], [0], None, [256], [0, 256])
blurred_cdf = blurred_hist.cumsum() / blurred_hist.sum()
cdf_normalized = blurred_cdf * 255 / blurred_cdf[-1]

equalized = np.interp(blurred.flatten(), range(256), cdf_normalized).reshape(img.shape).astype(np.uint8)
cv2.imwrite("images/equalized.png", equalized)

equalized_hist = cv2.calcHist([equalized], [0], None, [256], [0, 256])
plt.figure()
plt.plot(equalized_hist)
plt.title("Histogram After Equalisation")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.savefig("images/equalized_histogram.png")
plt.close()
