import os
import xml.etree.ElementTree as et
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train")

image_folder = os.path.join(dataset_path, "images")
annotation_folder = os.path.join(dataset_path, "labels")


image_files = sorted(
    filename
    for filename in os.listdir(image_folder)
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
)
if not image_files:
    raise SystemExit(f"No images found in: {image_folder}")

image_path = os.path.join(image_folder, image_files[0])
image = cv.imread(image_path)

if image is None:
    raise SystemExit(f"Could not read image: {image_path}")

# 1. Resize (YOLO input size)
resized = cv.resize(image, (640, 640))

# 2. Noise reduction
denoised = cv.GaussianBlur(resized, (5, 5), 0)

# 3. Contrast enhancement (CLAHE on the brightness channel)
lab = cv.cvtColor(denoised, cv.COLOR_BGR2LAB)
l, a, b = cv.split(lab)
l = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
enhanced = cv.cvtColor(cv.merge((l, a, b)), cv.COLOR_LAB2BGR)

# 4. Normalize pixel values to 0-1
normalized = enhanced.astype(np.float32) / 255.0

print("Original shape  :", image.shape)
print("Resized shape   :", resized.shape)
print("Pixel range after normalizing:", normalized.min(), "to", normalized.max())

# 5. Show before and after
steps = [image, resized, denoised, enhanced]
titles = ["Original", "Resized 640x640", "Denoised", "Enhanced"]

plt.figure(figsize=(14, 4))
for i, (im, t) in enumerate(zip(steps, titles)):
    plt.subplot(1, 4, i + 1)
    plt.imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    plt.title(t)
    plt.axis("off")
plt.tight_layout()
plt.savefig("preprocessing_steps.png")
plt.show()