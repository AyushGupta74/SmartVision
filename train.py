import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import build_model

tf.random.set_seed(42)
np.random.seed(42)

# ----------------------------------------------------
# Create folders
# ----------------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

TRAIN_DIR = "dataset/seg_train"
TEST_DIR = "dataset/seg_test"

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(128,128),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(128,128),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)
# ----------------------------------------------------
# class_names
# ----------------------------------------------------

class_names = train_dataset.class_names

for images, labels in train_dataset.take(1):
    print("First batch labels:")
    print(np.argmax(labels.numpy(), axis=1)[:10])

# ----------------------------------------------------
# Save class names for prediction
# ----------------------------------------------------
with open("models/classes.json", "w") as f:
    json.dump(class_names, f)

# ----------------------------------------------------
# Improve Performance
# ----------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = (
    train_dataset
    .cache()
    .shuffle(1000)
    .prefetch(AUTOTUNE)
)

test_dataset = (
    test_dataset
    .cache()
    .prefetch(AUTOTUNE)
)

# ----------------------------------------------------
# Build Model
# ----------------------------------------------------
model = build_model()

# ----------------------------------------------------
# Callbacks
# ----------------------------------------------------
callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        "models/cnn_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1
    )

]

# ----------------------------------------------------
# Train
# ----------------------------------------------------
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10,
    callbacks=callbacks
)

# ----------------------------------------------------
# Evaluate
# ----------------------------------------------------
loss, accuracy = model.evaluate(test_dataset)
print("\nChecking predictions on first test batch...")

for images, labels in test_dataset.take(1):

    predictions = model.predict(images)

    print("Predicted:")
    print(np.argmax(predictions, axis=1)[:20])

    print("Actual:")
    print(np.argmax(labels.numpy(), axis=1)[:20])

    break

print(f"\nTest Accuracy : {accuracy:.4f}")
print(f"Test Loss     : {loss:.4f}")


# ----------------------------------------------------
# Accuracy Graph
# ----------------------------------------------------
plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/accuracy.png")
plt.close()


# ----------------------------------------------------
# Loss Graph
# ----------------------------------------------------
plt.figure(figsize=(6,4))
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/loss.png")
plt.close()


# ----------------------------------------------------
# Confusion Matrix (OPTIMIZED)
# ----------------------------------------------------
print("\nGenerating Confusion Matrix...")

# OPTIMIZATION: Predict the entire dataset at once. 
# This is drastically faster than looping through batches.
predictions = model.predict(test_dataset)
pred_labels = np.argmax(predictions, axis=1)

# Extract the true labels from the dataset
# We concatenate them to match the shape of our predictions
true_labels_one_hot = np.concatenate([labels for images, labels in test_dataset], axis=0)
true_labels = np.argmax(true_labels_one_hot, axis=1)

cm = confusion_matrix(true_labels, pred_labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(8,8))

disp.plot(
    cmap="Blues",
    ax=ax,
    xticks_rotation=45
)
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()


# ----------------------------------------------------
# Final Summary
# ----------------------------------------------------

print("="*50)
print("Training Completed Successfully")
print("="*50)

print(f"Final Test Accuracy : {accuracy:.4f}")

print("Saved Model : models/cnn_model.keras")

print("Accuracy Graph : outputs/accuracy.png")

print("Loss Graph : outputs/loss.png")

print("Confusion Matrix : outputs/confusion_matrix.png")

# ----------------------------------------------------
# Save Training History
# ----------------------------------------------------
history_dict = history.history

np.save(
    "outputs/history.npy",
    history_dict
)