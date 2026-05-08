# Task 58 — Feedforward Neural Network
# Dataset: Pima Indians Diabetes (generated equivalent)
# Goal: Binary classification — predict diabetes (0/1)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

print("=" * 55)
print("Task 58 — Feedforward Neural Network")
print("=" * 55)

print("""
── What is a Feedforward Network? ──
In a feedforward network, information flows in ONE direction:
  Input → Hidden Layers → Output
There are NO loops or cycles (unlike RNNs).
Every layer feeds the next. It's the most basic deep learning
architecture and the foundation of all neural networks.

Signal flow: x → [W·x + b] → activation → next layer
""")

# ── Step 1: Generate Diabetes-like Dataset ───────────────────────────────────
np.random.seed(42)
n = 768

# 8 features: glucose, bmi, age, blood_pressure, skin_thickness,
#             insulin, pregnancies, pedigree
X = np.column_stack([
    np.random.normal(120, 30, n),   # glucose
    np.random.normal(32, 7, n),     # bmi
    np.random.normal(33, 11, n),    # age
    np.random.normal(70, 18, n),    # blood pressure
    np.random.normal(20, 15, n),    # skin thickness
    np.random.normal(80, 100, n),   # insulin
    np.random.randint(0, 10, n),    # pregnancies
    np.random.uniform(0.1, 2.4, n)  # pedigree
])

# Diabetes probability increases with glucose and bmi
prob = 1 / (1 + np.exp(-(-5 + 0.03 * X[:, 0] + 0.05 * X[:, 1])))
y = (np.random.uniform(0, 1, n) < prob).astype(int)

feature_names = ["Glucose", "BMI", "Age", "BloodPressure",
                 "SkinThickness", "Insulin", "Pregnancies", "Pedigree"]

print(f"Dataset: {n} patients, {X.shape[1]} features")
print(f"Diabetic (1): {y.sum()} | Non-diabetic (0): {(y==0).sum()}")

# ── Step 2: Preprocess ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Step 3: Build Feedforward Network ────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=(8,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8,  activation="relu"),
    keras.layers.Dense(1,  activation="sigmoid")   # binary output
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nFeedforward Network Architecture:")
model.summary()

# ── Step 4: Train ────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_split=0.15,
    verbose=0
)
print("\nTraining complete.")

# ── Step 5: Evaluate ─────────────────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()

print(f"\n── Evaluation ──")
print(f"Test Accuracy : {accuracy * 100:.2f}%")
print(f"Test Loss     : {loss:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

# ── Step 6: Plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Loss per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.suptitle("Task 58 — Feedforward Neural Network (Diabetes)", fontsize=13)
plt.tight_layout()
plt.savefig("task_58_feedforward_nn.png", dpi=150)
print("Plot saved as: task_58_feedforward_nn.png")