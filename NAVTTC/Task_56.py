# Task 56 — Neural Networks Demonstration (from scratch + Keras)
# Goal: Understand what a neural network is, then build one with Keras
# Dataset: XOR problem (from scratch) + Iris (with Keras)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow import keras

# ─────────────────────────────────────────────────────────────────
# PART 1: Neural Network concepts — XOR problem from scratch
# ─────────────────────────────────────────────────────────────────
print("=" * 55)
print("Task 56 — Neural Networks Demonstration")
print("=" * 55)

print("""
── What is a Neural Network? ──
A Neural Network is a series of layers:
  Input Layer  → receives raw features
  Hidden Layer → learns patterns via weights & activation
  Output Layer → produces the final prediction

Each connection has a WEIGHT. Training adjusts weights
using BACKPROPAGATION to minimise the LOSS function.
""")

# XOR truth table — not linearly separable
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

print("── XOR Problem (cannot be solved with a single line) ──")
print("Input → Expected Output")
for x, y in zip(X_xor, y_xor):
    print(f"  {x} → {y[0]}")

# Build a tiny neural net for XOR using Keras
model_xor = keras.Sequential([
    keras.layers.Dense(4, activation="relu", input_shape=(2,)),
    keras.layers.Dense(1, activation="sigmoid")
])
model_xor.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model_xor.fit(X_xor, y_xor, epochs=500, verbose=0)

predictions = (model_xor.predict(X_xor, verbose=0) > 0.5).astype(int)
print("\nXOR Neural Network Predictions:")
for x, pred, true in zip(X_xor, predictions, y_xor):
    status = "✓" if pred[0] == true[0] else "✗"
    print(f"  {x} → Predicted: {pred[0]}  True: {true[0]}  {status}")

# ─────────────────────────────────────────────────────────────────
# PART 2: Full Neural Network on Iris dataset
# ─────────────────────────────────────────────────────────────────
print("\n── Full Neural Network on Iris Dataset ──")

iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode labels: [0] → [1,0,0], [1] → [0,1,0], [2] → [0,0,1]
enc = OneHotEncoder(sparse_output=False)
y_encoded = enc.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Build neural network
model = keras.Sequential([
    keras.layers.Dense(16, activation="relu", input_shape=(4,)),
    keras.layers.Dense(8,  activation="relu"),
    keras.layers.Dense(3,  activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("\nModel Architecture:")
model.summary()

history = model.fit(X_train, y_train, epochs=100, validation_split=0.1, verbose=0)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy : {accuracy * 100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.suptitle("Task 56 — Neural Network Training (Iris)", fontsize=13)
plt.tight_layout()
plt.savefig("task_56_neural_network.png", dpi=150)
print("Plot saved as: task_56_neural_network.png")