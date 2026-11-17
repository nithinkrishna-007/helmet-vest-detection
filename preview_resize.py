import os
import cv2 as cv
import matplotlib.pyplot as plt

dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train")

image_folder = os.path.join(dataset_path, "images")
annotation_folder = os.path.join(dataset_path, "labels")

# Load a sample image
sample_image = os.path.join(image_folder, os.listdir(image_folder)[0])
img = cv.cvtColor(cv.imread(sample_image), cv.COLOR_BGR2RGB)

resized = cv.resize(img, (640, 640))
normalized = resized / 255.0      # pixel values now 0 to 1

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, im, t in zip(ax, [img, resized, normalized],
                    ["Original", "Resized 640x640", "Normalized"]):
    a.imshow(im)
    a.set_title(t)
    a.axis("off")
plt.savefig("preprocessing.png")
plt.show()