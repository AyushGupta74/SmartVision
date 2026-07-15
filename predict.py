import numpy as np

# We will pass the model as an argument from app.py.

def predict(model, classes, processed_image):
    # 1. Get raw probabilities from the model
    probabilities = model.predict(
        processed_image, 
        verbose=0
    )[0]

    # 2. Find the index of the highest probability
    predicted_idx = np.argmax(probabilities)

    # 3. Calculate confidence percentage
    confidence = probabilities[predicted_idx] * 100

    # 4. Get the indices of the top 3 predictions
    top3_indices = np.argsort(probabilities)[::-1][:3]

    return (
        classes[predicted_idx], 
        confidence, 
        probabilities, 
        top3_indices,
        classes
    )