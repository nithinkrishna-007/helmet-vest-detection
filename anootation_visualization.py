import os
import xml.etree.ElementTree as et
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import random

dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train")

image_folder = os.path.join(dataset_path, "images")
annotation_folder = os.path.join(dataset_path, "labels")

IMG_EXTS = (".jpg", ".jpeg", ".png")
NUM_SAMPLES = 9
CLASS_NAMES = ["Safety-Helmet", "Safety-Jacket"]

# one color (RGB) per class
COLORS = [(0, 200, 0), (255, 140, 0), (0, 120, 255), (220, 0, 220),
          (255, 0, 0), (0, 200, 200), (150, 150, 0), (120, 0, 200)]

# only use images that have a label file
images = [f for f in os.listdir(image_folder)
          if f.lower().endswith(IMG_EXTS)
          and os.path.exists(os.path.join(annotation_folder, os.path.splitext(f)[0] + ".txt"))]
print("Images with labels:", len(images))

samples = random.sample(images, min(NUM_SAMPLES, len(images)))

cols = 3
rows = (len(samples) + cols - 1) // cols
plt.figure(figsize=(5 * cols, 4 * rows))

for i, name in enumerate(samples):
    img = cv.imread(os.path.join(image_folder, name))
    if img is None:
        continue
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    label_path = os.path.join(annotation_folder, os.path.splitext(name)[0] + ".txt")
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) != 5:
                continue
            cls = int(p[0])
            xc, yc, bw, bh = map(float, p[1:])

            # relative (0-1) -> pixel coordinates
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            color = COLORS[cls % len(COLORS)]
            label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)

            thickness = max(2, w // 300)
            cv.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            cv.putText(img, label, (x1, max(y1 - 8, 15)),
                       cv.FONT_HERSHEY_SIMPLEX, max(0.5, w / 1200), color, 2)

    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.title(name, fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig("annotation_visualization.png")
plt.show()