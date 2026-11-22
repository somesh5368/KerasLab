# File: data_loader.py
# Purpose: Load and preprocess image data with robust path checks and demo generator

import os
import cv2
import numpy as np
from pyfiles.constants import IMG_SIZE

def generate_demo_dataset(base_dir="dataset", num_train=30, num_test=10, size=IMG_SIZE):
    """
    Generates a synthetic demo dataset of cats and dogs with distinct visual patterns
    so the application works out-of-the-box without requiring gigabytes of external data.
    """
    base_train_cats = os.path.join(base_dir, "training_set", "training_set", "cats")
    base_train_dogs = os.path.join(base_dir, "training_set", "training_set", "dogs")
    base_test_cats = os.path.join(base_dir, "test_set", "test_set", "cats")
    base_test_dogs = os.path.join(base_dir, "test_set", "test_set", "dogs")
    images_dir = "images"

    for d in [base_train_cats, base_train_dogs, base_test_cats, base_test_dogs, images_dir]:
        os.makedirs(d, exist_ok=True)

    def draw_animal(label, idx):
        img = np.full((size, size, 3), 245, dtype=np.uint8)
        cx, cy = size // 2, int(size * 0.55)
        r = int(size * 0.25)

        if label == "cat":
            # Cat face (gray)
            cv2.circle(img, (cx, cy), r, (180, 180, 180), -1)
            # Pointy triangle ears
            p1 = np.array([[cx - int(r*0.8), cy - int(r*0.4)], [cx - int(r*1.1), cy - int(r*1.4)], [cx - int(r*0.3), cy - int(r*0.8)]], np.int32)
            p2 = np.array([[cx + int(r*0.8), cy - int(r*0.4)], [cx + int(r*1.1), cy - int(r*1.4)], [cx + int(r*0.3), cy - int(r*0.8)]], np.int32)
            cv2.fillPoly(img, [p1], (140, 140, 140))
            cv2.fillPoly(img, [p2], (140, 140, 140))
            # Eyes (amber/cyan)
            cv2.circle(img, (cx - int(r*0.4), cy - int(r*0.2)), max(2, int(r*0.12)), (30, 190, 190), -1)
            cv2.circle(img, (cx + int(r*0.4), cy - int(r*0.2)), max(2, int(r*0.12)), (30, 190, 190), -1)
            # Nose
            cv2.circle(img, (cx, cy + int(r*0.2)), max(2, int(r*0.08)), (120, 120, 230), -1)
            # Whiskers
            cv2.line(img, (cx - int(r*0.9), cy + int(r*0.1)), (cx - int(r*0.3), cy + int(r*0.2)), (60, 60, 60), 1)
            cv2.line(img, (cx - int(r*0.9), cy + int(r*0.3)), (cx - int(r*0.3), cy + int(r*0.3)), (60, 60, 60), 1)
            cv2.line(img, (cx + int(r*0.9), cy + int(r*0.1)), (cx + int(r*0.3), cy + int(r*0.2)), (60, 60, 60), 1)
            cv2.line(img, (cx + int(r*0.9), cy + int(r*0.3)), (cx + int(r*0.3), cy + int(r*0.3)), (60, 60, 60), 1)
        else:
            # Dog face (warm brown/tan)
            cv2.circle(img, (cx, cy), int(r*1.05), (130, 170, 210), -1)
            # Floppy ears
            cv2.ellipse(img, (cx - int(r*0.9), cy - int(r*0.1)), (int(r*0.35), int(r*0.65)), 25, 0, 360, (90, 130, 170), -1)
            cv2.ellipse(img, (cx + int(r*0.9), cy - int(r*0.1)), (int(r*0.35), int(r*0.65)), -25, 0, 360, (90, 130, 170), -1)
            # Snout & big dark nose
            cv2.ellipse(img, (cx, cy + int(r*0.25)), (int(r*0.45), int(r*0.3)), 0, 0, 360, (170, 200, 230), -1)
            cv2.circle(img, (cx, cy + int(r*0.18)), max(3, int(r*0.15)), (25, 25, 25), -1)
            # Eyes
            cv2.circle(img, (cx - int(r*0.4), cy - int(r*0.2)), max(2, int(r*0.13)), (30, 30, 30), -1)
            cv2.circle(img, (cx + int(r*0.4), cy - int(r*0.2)), max(2, int(r*0.13)), (30, 30, 30), -1)

        # Subtle noise for diversity
        noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img

    for i in range(num_train):
        cv2.imwrite(os.path.join(base_train_cats, f"cat_{i+1}.jpg"), draw_animal("cat", i))
        cv2.imwrite(os.path.join(base_train_dogs, f"dog_{i+1}.jpg"), draw_animal("dog", i))

    for i in range(num_test):
        cv2.imwrite(os.path.join(base_test_cats, f"cat_{i+1}.jpg"), draw_animal("cat", 100+i))
        cv2.imwrite(os.path.join(base_test_dogs, f"dog_{i+1}.jpg"), draw_animal("dog", 100+i))

    cv2.imwrite(os.path.join(images_dir, "cat1.jpg"), draw_animal("cat", 999))
    cv2.imwrite(os.path.join(images_dir, "dog1.jpg"), draw_animal("dog", 999))


def _is_dir_empty(path):
    try:
        with os.scandir(path) as it:
            return next(it, None) is None
    except Exception:
        return True

def check_dataset_status(train_path, test_path):
    """Check if training and testing folders exist and have images."""
    for p in [train_path, test_path]:
        if not os.path.exists(p):
            return False, f"Directory '{p}' does not exist."
        cats_p = os.path.join(p, "cats")
        dogs_p = os.path.join(p, "dogs")
        if not (os.path.exists(cats_p) and os.path.exists(dogs_p)):
            return False, f"Subdirectories 'cats' and 'dogs' not found inside '{p}'."
        if _is_dir_empty(cats_p) or _is_dir_empty(dogs_p):
            return False, f"Folders inside '{p}' are empty."
    return True, "Dataset ready"


def load_data(path, img_size=IMG_SIZE):
    """
    Load and preprocess image data in RGB from the specified directory.
    Returns (x, y) arrays, with x shaped (-1, img_size, img_size, 3).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path '{path}' does not exist. Please generate demo dataset or provide a valid path.")

    cats_path = os.path.join(path, "cats")
    dogs_path = os.path.join(path, "dogs")

    if not os.path.exists(cats_path) or not os.path.exists(dogs_path):
        raise FileNotFoundError(f"Subdirectories 'cats' and 'dogs' required inside '{path}'.")

    x_dataset = []
    y_dataset = []

    # Process cats (label 1)
    for img_name in os.listdir(cats_path):
        _process_file(os.path.join(cats_path, img_name), x_dataset, y_dataset, 1, img_size)

    # Process dogs (label 0)
    for img_name in os.listdir(dogs_path):
        _process_file(os.path.join(dogs_path, img_name), x_dataset, y_dataset, 0, img_size)

    if len(x_dataset) == 0:
        raise ValueError(f"No valid images could be loaded from '{path}'.")

    x = np.array(x_dataset, dtype="float32") / 255.0
    y = np.array(y_dataset, dtype="float32").reshape(-1, 1)

    # Shuffle
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    return x[indices], y[indices]


def _process_file(file_path, x_dataset, y_dataset, label, img_size):
    try:
        # Load image as RGB (3 channels) for CNN / VGG / ResNet compatibility
        bgr = cv2.imread(file_path)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (img_size, img_size))
            x_dataset.append(resized)
            y_dataset.append(label)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
