import os
import xml.etree.ElementTree as et
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt


dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train")

image_folder = os.path.join(dataset_path, "images")
annotation_folder = os.path.join(dataset_path, "labels")

print("images folder exists:", os.path.exists(image_folder))
print("annotation folder exists:", os.path.exists(annotation_folder))

# to view data
CLASS_NAMES = ["Safety-Helmet", "Safety-Jacket"]
image_name = "helmet_jacket_00003.jpg"

image_path = os.path.join(image_folder, image_name)
label_path = os.path.join(annotation_folder, os.path.splitext(image_name)[0] + ".txt")

print("Running from:", os.getcwd())
print("image exists:", os.path.exists(image_path))
print("label exists:", os.path.exists(label_path))

if not (os.path.exists(image_path) and os.path.exists(label_path)):
    print("Items here:", os.listdir("."))
    raise SystemExit("Fix the paths at the top and run again.")

image = cv.imread(image_path)
if image is None:
    raise SystemExit(f"Could not read image: {image_path}")
h, w = image.shape[:2]
print("Image shape:", image.shape)

with open(label_path) as f:
    for line in f:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])

        # convert from relative (0-1) to pixel coordinates
        xmin = int((xc - bw / 2) * w)
        ymin = int((yc - bh / 2) * h)
        xmax = int((xc + bw / 2) * w)
        ymax = int((yc + bh / 2) * h)

        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv.putText(image, name, (xmin, max(ymin - 10, 15)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        print("Found:", name, (xmin, ymin, xmax, ymax))

image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.figure(figsize=(10, 7))
plt.imshow(image_rgb)
plt.axis("off")
plt.savefig("output.png")
plt.show()


# Dataset checks
IMG_EXTS = (".jpg", ".jpeg", ".png")
image_files = [
    f for f in os.listdir(image_folder)
    if os.path.splitext(f)[1].lower() in IMG_EXTS
]
label_names = {
    os.path.splitext(f)[0]
    for f in os.listdir(annotation_folder)
    if f.lower().endswith(".txt")
}

corrupt_images = []
images_without_labels = []
bad_labels = set()
class_counts = {}
image_sizes = {}
image_names = set()

for filename in image_files:
    image_name_without_ext = os.path.splitext(filename)[0]
    image_names.add(image_name_without_ext)
    image = cv.imread(os.path.join(image_folder, filename))
    if image is None:
        corrupt_images.append(filename)
        continue

    height, width = image.shape[:2]
    size = (width, height)
    image_sizes[size] = image_sizes.get(size, 0) + 1
    if image_name_without_ext not in label_names:
        images_without_labels.append(filename)

for label_name in label_names:
    label_path = os.path.join(annotation_folder, label_name + ".txt")
    with open(label_path, encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            values = line.split()
            if not values:
                continue
            try:
                class_id = int(values[0])
                box = list(map(float, values[1:]))
            except ValueError:
                bad_labels.add(f"{label_name}.txt:{line_number}")
                continue

            if (
                len(box) != 4
                or class_id < 0
                or class_id >= len(CLASS_NAMES)
                or not all(0 <= value <= 1 for value in box)
            ):
                bad_labels.add(f"{label_name}.txt:{line_number}")
                continue
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

labels_without_images = sorted(label_names - image_names)

print("\nDataset summary")
print("Images:", len(image_files))
print("Labels:", len(label_names))
print("Corrupt images:", len(corrupt_images), corrupt_images[:10])
print("Images without labels:", len(images_without_labels), images_without_labels[:10])
print("Labels without images:", len(labels_without_images), labels_without_images[:10])
print("Invalid label rows:", len(bad_labels), sorted(bad_labels)[:10])
print("Boxes per class:")
for class_id, count in sorted(class_counts.items()):
    print(f"  {CLASS_NAMES[class_id]}: {count}")
print("Most common image sizes (width, height):")
for size, count in sorted(image_sizes.items(), key=lambda item: -item[1])[:5]:
    print(f"  {size}: {count} images")