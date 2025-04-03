import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("images/checkerboard.png", cv2.IMREAD_GRAYSCALE)

# Histogram and CDF
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
cdf = hist.cumsum() / hist.sum()
plt.plot(cdf)
plt.title("CDF")
plt.savefig("images/cdf.png")

# Gaussian Blur
blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
cv2.imwrite("images/blurred.png", blurred)

# FFT Low-pass Filtering
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros_like(img)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
filtered_fshift = fshift * mask
f_ishift = np.fft.ifftshift(filtered_fshift)
filtered_img = np.fft.ifft2(f_ishift).real.astype(np.uint8)
cv2.imwrite("images/frequency_filtered.png", filtered_img)
