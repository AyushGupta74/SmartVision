import cv2
import numpy as np

IMG_SIZE = 128


def preprocess_image(uploaded_file):

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

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = image.astype("float32") / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return original, image