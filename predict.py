import tensorflow as tf
import numpy as np
import json

model = tf.keras.models.load_model(
    "models/cnn_model.keras"
)

print("Loaded model from: models/cnn_model.keras")
print("Model input:", model.input_shape)

with open("models/classes.json") as f:
    classes = json.load(f)
print("Loaded Classes:", classes)

def predict(processed_image):

    print("Input Shape:", processed_image.shape)

    probabilities = model.predict(
        processed_image,
        verbose=0
    )[0]

    print("\nProbabilities:")
    for cls, prob in zip(classes, probabilities):
        print(f"{cls:10s}: {prob:.4f}")

    print("Probabilities:", probabilities)

    predicted = np.argmax(probabilities)

    confidence = probabilities[predicted] * 100

    top3 = np.argsort(probabilities)[::-1][:3]

    return (
        classes[predicted],
        confidence,
        probabilities,
        top3,
        classes
    )
