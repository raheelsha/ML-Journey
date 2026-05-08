# Task 64 — CNN on CIFAR-10 Dataset
# Dataset: CIFAR-10 (60,000 colour images, 10 classes)
# Goal: Classify images into 10 categories using a deep CNN

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print("=" * 55)
print("Task 64 — CNN on CIFAR-10 Dataset")
print("=" * 55)

print("""
── CIFAR-10 Dataset ──
60,000 colour images, 32x32 pixels, 3 channels (RGB)
10 classes: airplane, automobile, bird, cat, deer,
            dog, frog, horse, ship, truck
Train: 50,000 images | Test: 10,000 images

This is much harder than MNIST — colour, complex shapes,
varied backgrounds — so we use a deeper CNN + data augmentation.
""")

# ── Step 1: Load & Preprocess CIFAR-10 ──────────────────────────────────────
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

class_names = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]

# Normalise pixel values to [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0

y_train_enc = keras.utils.to_categorical(y_train, 10)
y_test_enc  = keras.utils.to_categorical(y_test,  10)

print(f"Train shape : {X_train.shape}")
print(f"Test shape  : {X_test.shape}")

# ── Step 2: Visualise Sample Images ─────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i])
    ax.set_title(class_names[y_train[i][0]], fontsize=9)
    ax.axis("off")
plt.suptitle("Task 64 — CIFAR-10 Sample Images", fontsize=13)
plt.tight_layout()
plt.savefig("task_64_cifar10_samples.png", dpi=150)
print("Sample images saved as: task_64_cifar10_samples.png")

# ── Step 3: Build Deep CNN ───────────────────────────────────────────────────
model = keras.Sequential([
    # Block 1
    keras.layers.Conv2D(32, (3,3), activation="relu", padding="same",
                        input_shape=(32,32,3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.25),

    # Block 2
    keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.25),

    # Block 3
    keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.3),

    # Classifier
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Architecture:")
model.summary()

# ── Step 4: Train ────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train_enc,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# ── Step 5: Evaluate ─────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test_enc, verbose=0)
print(f"\nTest Accuracy : {accuracy * 100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

# ── Step 6: Per-class accuracy ───────────────────────────────────────────────
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = y_test.flatten()
print("\nPer-class Accuracy:")
for i, cls in enumerate(class_names):
    mask = y_true == i
    cls_acc = np.mean(y_pred[mask] == y_true[mask])
    bar = "█" * int(cls_acc * 20)
    print(f"  {cls:<12}: {bar:<20} {cls_acc*100:.1f}%")

# ── Step 7: Predictions on test images ──────────────────────────────────────
fig2, axes2 = plt.subplots(3, 5, figsize=(13, 8))
for i, ax in enumerate(axes2.flat):
    ax.imshow(X_test[i])
    pred = y_pred[i]
    true = y_true[i]
    color = "green" if pred == true else "red"
    ax.set_title(f"P:{class_names[pred]}\nT:{class_names[true]}",
                 color=color, fontsize=7)
    ax.axis("off")
plt.suptitle("Task 64 — CNN CIFAR-10 Predictions (Green=Correct)", fontsize=12)
plt.tight_layout()
plt.savefig("task_64_cifar10_predictions.png", dpi=150)
print("Predictions saved as: task_64_cifar10_predictions.png")

# ── Step 8: Training history ─────────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4))
axes3[0].plot(history.history["accuracy"],     label="Train")
axes3[0].plot(history.history["val_accuracy"], label="Validation")
axes3[0].set_title("Accuracy per Epoch")
axes3[0].set_xlabel("Epoch")
axes3[0].legend()

axes3[1].plot(history.history["loss"],     label="Train")
axes3[1].plot(history.history["val_loss"], label="Validation")
axes3[1].set_title("Loss per Epoch")
axes3[1].set_xlabel("Epoch")
axes3[1].legend()

plt.suptitle("Task 64 — CIFAR-10 CNN Training History", fontsize=13)
plt.tight_layout()
plt.savefig("task_64_cifar10_training.png", dpi=150)
print("Training history saved as: task_64_cifar10_training.png")