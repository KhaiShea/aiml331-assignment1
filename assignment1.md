# Assignment 1 – AIML331 2025  
**Name:** Khai Dye-Brinkman  
**Student ID:** 300550065
**Date:** April 03, 2025  
**Code Repository:** [https://github.com/yourusername/aiml331-assignment1](https://github.com/yourusername/aiml331-assignment1)

---

## 1. Camera Problem

### 1.1 [R, t] Matrix

> Your pinhole camera is at [X, Y, Z] = [0, 0, −10] and sits on a horizontal table (X and Z specify the horizontal plane). It is pointing 30 degrees to the right relative to the origin of the world coordinates. Compute the [R, t] matrix converting world coordinates to camera coordinates. [5 marks]

```python
import numpy as np

theta = np.radians(30)
R = np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)]
])
C = np.array([0, 0, -10])
t = -R @ C
Rt = np.column_stack((R, t))
print("R =\n", R)
print("t =\n", t)
print("[R|t] =\n", Rt)
```

### 1.2 Line in Projective Plane

> Your camera has f = 0.1 (focal length, as used in pinhole camera). Using homogeneous coordinates compute the equation of a line on the projective plane that goes through the points that correspond to [0, 1, 0] and [0, 0, 1] in the world coordinates. [10 marks]

```python
f = 0.1
K = np.array([
    [f, 0, 0],
    [0, f, 0],
    [0, 0, 1]
])
X1 = np.array([0, 1, 0, 1])
X2 = np.array([0, 0, 1, 1])
P1 = K @ Rt @ X1
P2 = K @ Rt @ X2
P1 /= P1[2]
P2 /= P2[2]
line = np.cross(P1, P2)
print("Line (homogeneous):", line)
```

---

## 2. Checkerboard Problem

### 2.1 Find Checkerboard Image

> Find an image on-line that displays a checkerboard (provide source website). [1 mark]

Image: https://upload.wikimedia.org/wikipedia/commons/3/3c/Checkerboard_pattern.png

### 2.2 Histogram and Cumulative Probability Function

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('checkerboard.png', cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
cdf = hist.cumsum() / hist.sum()
plt.plot(cdf)
plt.title("CDF")
plt.show()
```

### 2.3 Spatial Low-pass Filter

```python
blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
```

### 2.4 Separable Filter?

Yes — the Gaussian filter is separable.

### 2.5 Frequency Domain Low-pass Filter

```python
import numpy as np

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros_like(img)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
filtered_fshift = fshift * mask
f_ishift = np.fft.ifftshift(filtered_fshift)
filtered_img = np.fft.ifft2(f_ishift).real.astype(np.uint8)
```

### 2.6 Histogram of Filtered Image

```python
filtered_hist = cv2.calcHist([filtered_img], [0], None, [256], [0, 256])
plt.plot(filtered_hist)
plt.title("Histogram of Filtered Image")
plt.show()
```

### 2.7 Histogram Equalisation using CDF

```python
cdf_normalized = cdf * 255 / cdf[-1]
equalized = np.interp(blurred.flatten(), range(256), cdf_normalized).reshape(img.shape)
```

---

## ✅ Notes

- Camera transforms used rotation + homogeneous translation.
- Filtering used both spatial and Fourier domains.
- Histograms and equalisation implemented via CDF mapping.
- Code, images, and all scripts are in the GitHub repository.
