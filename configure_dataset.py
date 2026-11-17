import os

dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train")

image_folder = os.path.join(dataset_path, "images")
annotation_folder = os.path.join(dataset_path, "labels")
PROJECT = os.getcwd()
SKIP = {"venv", "runs", ".git", "__pycache__"}
IMG_EXTS = (".jpg", ".jpeg", ".png")
NUM_SAMPLES = 9
CLASS_NAMES = ["Safety-Helmet", "Safety-Jacket"]

# 1. Find the folder that contains train/images
dataset_root = None
for dirpath, dirnames, _ in os.walk(PROJECT):
    dirnames[:] = [d for d in dirnames if d not in SKIP]
    if os.path.isdir(os.path.join(dirpath, "train", "images")):
        dataset_root = dirpath
        break

if dataset_root is None:
    print("Could not find a 'train/images' folder inside:", PROJECT)
    print("Items here:", os.listdir(PROJECT))
    raise SystemExit("Copy the extracted dataset folder into the project and run again.")

print("Dataset folder found:", dataset_root)

# 2. Check each split
splits = {}
for key, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
    img_dir = os.path.join(dataset_root, folder, "images")
    lbl_dir = os.path.join(dataset_root, folder, "labels")
    if not os.path.isdir(img_dir):
        print(f"{folder}: not found (skipped)")
        continue
    imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)]
    lbls = [f for f in os.listdir(lbl_dir)] if os.path.isdir(lbl_dir) else []
    print(f"{folder}: {len(imgs)} images, {len(lbls)} label files")
    splits[key] = f"{folder}/images"

if "train" not in splits or "val" not in splits:
    raise SystemExit("train and valid folders are both needed for training.")

# 3. Check class ids in the train labels
max_id = -1
lbl_dir = os.path.join(dataset_root, "train", "labels")
for f in os.listdir(lbl_dir):
    if f.endswith(".txt"):
        with open(os.path.join(lbl_dir, f)) as fh:
            for line in fh:
                p = line.split()
                if p:
                    max_id = max(max_id, int(p[0]))
print("Highest class id in train labels:", max_id, "| classes listed:", len(CLASS_NAMES))
if max_id >= len(CLASS_NAMES):
    raise SystemExit("Labels use more class ids than CLASS_NAMES. Fix CLASS_NAMES.")

# 4. Write data.yml with an absolute path
abs_path = dataset_root.replace("\\", "/")
with open("data.yml", "w") as f:
    f.write(f"path: {abs_path}\n")
    f.write(f"train: {splits['train']}\n")
    f.write(f"val: {splits['val']}\n")
    if "test" in splits:
        f.write(f"test: {splits['test']}\n")
    f.write(f"nc: {len(CLASS_NAMES)}\n")
    f.write("names:\n")
    for i, n in enumerate(CLASS_NAMES):
        f.write(f"  {i}: {n}\n")

print("\ndata.yml written. Contents:\n")
print(open("data.yml").read())