import cv2
import numpy as np

IMG_SIZE = 128


def preprocess_image(uploaded_file):

# 1. Read the raw bytes from the uploaded Streamlit file
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Invalid image")

    original = image.copy()

# 2. Fix OpenCV's BGR color quirk to standard RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

# 3. Resize to match the model's expected input
    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = image.astype("float32")

    image = np.expand_dims(
        image,
        axis=0
    )

    return original, image