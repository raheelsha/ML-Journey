# Task 59 — Loss Functions in Neural Networks
# Goal: Understand and demonstrate 7 key loss functions

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("=" * 55)
print("Task 59 — Loss Functions in Neural Networks")
print("=" * 55)

print("""
── What is a Loss Function? ──
A loss function measures how WRONG the model's predictions are.
During training, the optimizer tries to MINIMISE this value.
Choosing the right loss function depends on your task type.
""")

# ─────────────────────────────────────────
# 1. Mean Squared Error (MSE) — Regression
# ─────────────────────────────────────────
print("1. Mean Squared Error (MSE) — for Regression")
print("   Formula: MSE = mean((y_true - y_pred)²)")
y_true_reg = np.array([3.0, 5.0, 2.5, 7.0])
y_pred_reg = np.array([2.5, 5.0, 4.0, 8.0])
mse = np.mean((y_true_reg - y_pred_reg) ** 2)
print(f"   y_true: {y_true_reg} | y_pred: {y_pred_reg}")
print(f"   MSE = {mse:.4f}")

# ─────────────────────────────────────────
# 2. Mean Absolute Error (MAE) — Regression
# ─────────────────────────────────────────
print("\n2. Mean Absolute Error (MAE) — for Regression")
print("   Formula: MAE = mean(|y_true - y_pred|)")
mae = np.mean(np.abs(y_true_reg - y_pred_reg))
print(f"   MAE = {mae:.4f}")
print("   Less sensitive to outliers than MSE")

# ─────────────────────────────────────────
# 3. Binary Cross-Entropy — Binary Classification
# ─────────────────────────────────────────
print("\n3. Binary Cross-Entropy — Binary Classification")
print("   Formula: -mean(y·log(p) + (1-y)·log(1-p))")
y_true_bin = np.array([1, 0, 1, 1])
y_pred_bin = np.array([0.9, 0.1, 0.8, 0.4])
bce = -np.mean(
    y_true_bin * np.log(y_pred_bin + 1e-9) +
    (1 - y_true_bin) * np.log(1 - y_pred_bin + 1e-9)
)
print(f"   y_true: {y_true_bin} | y_pred: {y_pred_bin}")
print(f"   Binary Cross-Entropy = {bce:.4f}")

# ─────────────────────────────────────────
# 4. Categorical Cross-Entropy — Multi-class
# ─────────────────────────────────────────
print("\n4. Categorical Cross-Entropy — Multi-class Classification")
print("   Formula: -sum(y_true * log(y_pred))")
y_true_cat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred_cat = np.array([[0.9, 0.05, 0.05],
                       [0.1, 0.8,  0.1],
                       [0.2, 0.2,  0.6]])
cce = -np.mean(np.sum(y_true_cat * np.log(y_pred_cat + 1e-9), axis=1))
print(f"   Categorical Cross-Entropy = {cce:.4f}")

# ─────────────────────────────────────────
# 5. Hinge Loss — SVM-style classification
# ─────────────────────────────────────────
print("\n5. Hinge Loss — used in SVM / margin classifiers")
print("   Formula: mean(max(0, 1 - y_true * y_pred))")
y_true_hinge = np.array([1, -1, 1, -1])
y_pred_hinge = np.array([0.8, -0.6, -0.3, 0.5])
hinge = np.mean(np.maximum(0, 1 - y_true_hinge * y_pred_hinge))
print(f"   Hinge Loss = {hinge:.4f}")

# ─────────────────────────────────────────
# 6. Huber Loss — robust regression
# ─────────────────────────────────────────
print("\n6. Huber Loss — Regression, robust to outliers")
print("   Combines MSE (small errors) and MAE (large errors)")
delta = 1.0
errors = y_true_reg - y_pred_reg
huber = np.where(
    np.abs(errors) <= delta,
    0.5 * errors ** 2,
    delta * (np.abs(errors) - 0.5 * delta)
)
print(f"   Huber Loss = {np.mean(huber):.4f}")

# ─────────────────────────────────────────
# 7. Kullback-Leibler Divergence
# ─────────────────────────────────────────
print("\n7. KL Divergence — measures difference between two distributions")
print("   Formula: sum(p * log(p / q))")
p = np.array([0.4, 0.3, 0.3])
q = np.array([0.3, 0.4, 0.3])
kl_div = np.sum(p * np.log(p / q))
print(f"   P = {p} | Q = {q}")
print(f"   KL Divergence = {kl_div:.4f}")

# ── Summary Table ─────────────────────────────────────────────────────────────
print("\n── Loss Function Summary ──")
print(f"{'Loss Function':<30} {'Use Case':<35} {'Value'}")
print("-" * 75)
print(f"{'MSE':<30} {'Regression':<35} {mse:.4f}")
print(f"{'MAE':<30} {'Regression (outlier-robust)':<35} {mae:.4f}")
print(f"{'Binary Cross-Entropy':<30} {'Binary Classification':<35} {bce:.4f}")
print(f"{'Categorical Cross-Entropy':<30} {'Multi-class Classification':<35} {cce:.4f}")
print(f"{'Hinge Loss':<30} {'SVM / Margin classifiers':<35} {hinge:.4f}")
print(f"{'Huber Loss':<30} {'Regression w/ outliers':<35} {np.mean(huber):.4f}")
print(f"{'KL Divergence':<30} {'Distribution comparison':<35} {kl_div:.4f}")

# ── Visualise MSE vs MAE vs Huber ────────────────────────────────────────────
errors_range = np.linspace(-3, 3, 200)
mse_curve   = errors_range ** 2
mae_curve   = np.abs(errors_range)
huber_curve = np.where(np.abs(errors_range) <= 1,
                       0.5 * errors_range**2,
                       np.abs(errors_range) - 0.5)

plt.figure(figsize=(9, 5))
plt.plot(errors_range, mse_curve,   label="MSE",   linewidth=2)
plt.plot(errors_range, mae_curve,   label="MAE",   linewidth=2)
plt.plot(errors_range, huber_curve, label="Huber", linewidth=2, linestyle="--")
plt.xlabel("Prediction Error")
plt.ylabel("Loss Value")
plt.title("Task 59 — Loss Function Comparison: MSE vs MAE vs Huber")
plt.legend()
plt.ylim(0, 6)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("task_59_loss_functions.png", dpi=150)
print("\nPlot saved as: task_59_loss_functions.png")
