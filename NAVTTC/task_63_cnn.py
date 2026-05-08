# Task 63 — Convolutional Neural Network (CNN) Demonstration
# Dataset: MNIST (simple, fast to train — ideal for understanding CNNs)
# Goal: Understand CNN architecture and train an image classifier

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print("=" * 55)
print("Task 63 — Convolutional Neural Network (CNN)")
print("=" * 55)

print("""
── What is a CNN? ──
A CNN is designed specifically for IMAGE data.
Instead of connecting every pixel to every neuron (expensive),
CNN uses:
  Conv2D Layer   → learns local patterns (edges, shapes, textures)
                   via small sliding filters/kernels
  MaxPooling2D   → downsamples feature maps (reduces size)
  Flatten        → converts 2D feature maps to 1D vector
  Dense          → final classification layers

Architecture: Image → Conv → Pool → Conv → Pool → Flatten → Dense → Output
""")

# ── Step 1: Load & Preprocess MNIST ─────────────────────────────────────────
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Reshape to (samples, height, width, channels) — 1 channel = grayscale
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

y_train_enc = keras.utils.to_categorical(y_train, 10)
y_test_enc  = keras.utils.to_categorical(y_test,  10)

print(f"Train shape : {X_train.shape}")
print(f"Test shape  : {X_test.shape}")

# ── Step 2: Build CNN ────────────────────────────────────────────────────────
model = keras.Sequential([
    # Block 1: Conv + Pool
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Block 2: Conv + Pool
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Classifier head
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Architecture:")
model.summary()

# ── Step 3: Train ────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train_enc,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# ── Step 4: Evaluate ─────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test_enc, verbose=0)
print(f"\nTest Accuracy : {accuracy * 100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

# ── Step 5: Visualise Feature Maps ───────────────────────────────────────────
# Show what the first Conv layer "sees" for one image
sample_img = X_test[0:1]

# Get output of first conv layer (Keras 3 compatible)
inp = keras.Input(shape=(28, 28, 1))
conv_layer_model = keras.Model(
    inputs=inp,
    outputs=model.layers[0](inp)
)
feature_maps = conv_layer_model.predict(sample_img, verbose=0)

fig, axes = plt.subplots(4, 8, figsize=(14, 7))
for i, ax in enumerate(axes.flat):
    if i < feature_maps.shape[-1]:
        ax.imshow(feature_maps[0, :, :, i], cmap="viridis")
    ax.axis("off")
plt.suptitle("Task 63 — CNN: First Conv Layer Feature Maps (32 filters)", fontsize=12)
plt.tight_layout()
plt.savefig("task_63_cnn_feature_maps.png", dpi=150)
print("Feature maps saved as: task_63_cnn_feature_maps.png")

# ── Step 6: Sample predictions ───────────────────────────────────────────────
predictions = np.argmax(model.predict(X_test[:10], verbose=0), axis=1)
fig2, axes2 = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes2.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    pred, true = predictions[i], y_test[i]
    ax.set_title(f"P:{pred} T:{true}", color="green" if pred==true else "red", fontsize=9)
    ax.axis("off")
plt.suptitle("Task 63 — CNN MNIST Predictions", fontsize=12)
plt.tight_layout()
plt.savefig("task_63_cnn_predictions.png", dpi=150)
print("Predictions plot saved as: task_63_cnn_predictions.png")
