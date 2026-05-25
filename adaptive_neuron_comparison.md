# Optimization Strategies: Batch vs. Stochastic vs. Mini-Batch Gradient Descent

When training machine learning models (such as an Adaline or Neural Network), how you feed data into the model heavily influences memory usage, computational speed, and convergence stability. This document outlines the three primary data-batching paradigms.

---

## Quick Comparison Table

| Strategy | Batch Size | Operational Flow | Key Advantages | Key Disadvantages |
| :--- | :--- | :--- | :--- | :--- |
| **Batch Gradient Descent** | **The entire dataset** ($m$ samples) | Computes the network input and errors for *all* samples simultaneously via matrix operations (`X.T.dot(errors)`), then makes a single weight update per epoch. | • Stable, smooth convergence.<br>• Mathematically precise trajectory. | • Very memory intensive.<br>• Can get stuck easily in local minima/saddle points. |
| **Stochastic Gradient Descent (SGD)** | **Exactly 1 sample** | Loops through the dataset row-by-row. Evaluates a single sample ($x_i$), calculates its individual error, and immediately adjusts the weights before moving to the next row. | • Low memory footprint.<br>• Frequent updates allow online learning.<br>• High noise helps escape local minima. | • Noisy updates can cause the loss to oscillate.<br>• Cannot take advantage of modern GPU vectorization hardware. |
| **Mini-batch Gradient Descent** | **A user-defined subset** (e.g., 32, 64, 128) | Splits the dataset into small random groups. Aggregates the errors for that group, updates the weights, and then iterates to the next group. | • **The Industry Standard**.<br>• Best of both worlds: hardware accelerated (vectorized) yet computationally efficient.<br>• Smoother path than pure SGD. | • Introduces an extra hyperparameter to tune (Batch Size). |

---

## Core Characteristics Breakdown

### 1. Batch Gradient Descent
* **Behavior:** Updates weights once per epoch.
* **Math Matrix Shape:** Evaluates $X$ of shape `[m, n_features]` natively.
* **Best Used For:** Small-to-medium datasets that easily fit into RAM/VRAM.

### 2. Stochastic Gradient Descent (SGD)
* **Behavior:** Updates weights $m$ times per epoch.
* **Math Matrix Shape:** Loops sample-by-sample, squeezing data into a single 1D feature array `[n_features]` at each step.
* **Data Dependency Note:** Data **must be shuffled** every epoch before looping. If data contains sequential patterns (e.g., all class `0` instances clustered together), the step-by-step updates will cause severe optimization bias and cyclic updates.

### 3. Mini-batch Gradient Descent
* **Behavior:** Splits data into subsets of size $B$. Updates weights $m/B$ times per epoch.
* **Math Matrix Shape:** Evaluates sub-matrices of shape `[B, n_features]`.
* **Hardware Efficiency:** This strategy is heavily favored because modern linear algebra hardware (like NVIDIA GPUs and vectorized CPU instructions) optimizes parallel matrix multiplications on small chunks exceptionally well.