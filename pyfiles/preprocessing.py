# File: preprocessing.py
# Purpose: Preprocess images from file path or uploaded bytes for model inference

import cv2
import numpy as np
from PIL import Image
import io
from pyfiles.constants import IMG_SIZE

def preprocess_image(image_input, img_size=IMG_SIZE):
    """
    Preprocess a single image for prediction.
    Accepts:
      - File path (str)
      - Streamlit UploadedFile
      - PIL Image
      - bytes

    Returns:
        numpy.ndarray: Shape (1, img_size, img_size, 3) normalized to [0, 1].
    """
    rgb_image = load_display_image(image_input)
    resized = cv2.resize(rgb_image, (img_size, img_size))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)


def load_display_image(image_input):
    """
    Loads an image input into an RGB numpy array for display or processing.
    """
    if isinstance(image_input, str):
        bgr = cv2.imread(image_input)
        if bgr is None:
            raise ValueError(f"Could not read image from path '{image_input}'.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    elif hasattr(image_input, "read"):
        # Streamlit UploadedFile or file-like object
        image_bytes = image_input.read()
        if hasattr(image_input, "seek"):
            image_input.seek(0)  # Reset stream pointer
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil_img)

    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))

    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        return image_input

    else:
        raise ValueError("Unsupported image input format.")
