# Task 57 — MLP: Multi-Layer Perceptron with Keras
# Dataset: MNIST Handwritten Digits
# Goal: Classify digits 0-9 using a fully connected neural network

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print("=" * 55)
print("Task 57 — MLP: Multi-Layer Perceptron")
print("=" * 55)

print("""
── What is an MLP? ──
A Multi-Layer Perceptron (MLP) is a feedforward neural network
with one or more hidden layers. Each neuron in one layer connects
to every neuron in the next layer (fully connected / dense).
Architecture: Input → Dense → Dense → ... → Output
""")

# ── Step 1: Load MNIST Dataset ───────────────────────────────────────────────
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")
print(f"Image shape      : {X_train.shape[1:]} (28x28 pixels)")
print(f"Classes          : 0 through 9")

# ── Step 2: Preprocess ───────────────────────────────────────────────────────
# Flatten 28x28 → 784, and normalise pixel values from [0,255] to [0,1]
X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 784).astype("float32") / 255.0

# One-hot encode labels
y_train_enc = keras.utils.to_categorical(y_train, 10)
y_test_enc  = keras.utils.to_categorical(y_test,  10)

# ── Step 3: Build MLP ────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Dense(256, activation="relu", input_shape=(784,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64,  activation="relu"),
    keras.layers.Dense(10,  activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nMLP Architecture:")
model.summary()

# ── Step 4: Train ────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train_enc,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# ── Step 5: Evaluate ─────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test_enc, verbose=0)
print(f"\nTest Accuracy : {accuracy * 100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

# ── Step 6: Sample Predictions ───────────────────────────────────────────────
predictions = np.argmax(model.predict(X_test[:10], verbose=0), axis=1)
print(f"\nSample Predictions (first 10):")
print(f"  Predicted : {predictions}")
print(f"  Actual    : {y_test[:10]}")

# ── Step 7: Visualise ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    pred = predictions[i]
    true = y_test[i]
    color = "green" if pred == true else "red"
    ax.set_title(f"P:{pred} T:{true}", color=color, fontsize=9)
    ax.axis("off")
plt.suptitle("Task 57 — MLP Digit Predictions (Green=Correct)", fontsize=12)
plt.tight_layout()
plt.savefig("task_57_mlp_predictions.png", dpi=150)
print("Plot saved as: task_57_mlp_predictions.png")

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
axes2[0].plot(history.history["accuracy"],     label="Train")
axes2[0].plot(history.history["val_accuracy"], label="Validation")
axes2[0].set_title("Accuracy per Epoch")
axes2[0].set_xlabel("Epoch")
axes2[0].legend()

axes2[1].plot(history.history["loss"],     label="Train")
axes2[1].plot(history.history["val_loss"], label="Validation")
axes2[1].set_title("Loss per Epoch")
axes2[1].set_xlabel("Epoch")
axes2[1].legend()

plt.suptitle("Task 57 — MLP Training History (MNIST)", fontsize=13)
plt.tight_layout()
plt.savefig("task_57_mlp_training.png", dpi=150)
print("Training history saved as: task_57_mlp_training.png")
