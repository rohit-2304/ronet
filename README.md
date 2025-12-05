# Ronet: A NumPy-only Deep Learning Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)

**Ronet** is a lightweight, modular deep learning library built entirely from scratch using only `numpy`. 

It is designed for educational purposes, demystifying the "black box" of frameworks like PyTorch or TensorFlow. It implements backpropagation, optimizers, and layer logic using raw matrix operations, making it an excellent tool for understanding the mathematics of deep learning.

## 🚀 Features

* **No Heavy Dependencies:** Built strictly on NumPy.
* **Keras-like API:** Familiar `.add()`, `.compile()`, and `.train()` structure.
* **Modular Design:** easily extensible layers, optimizers, and loss functions.
* **Implements Core Concepts:**
    * **Layers:** Dense (Fully Connected), Dropout, BatchNorm.
    * **Activations:** ReLU, Softmax, Sigmoid, Linear.
    * **Optimizers:** Adam, RMSProp, SGD (with Momentum), Adagrad.
    * **Losses:** Categorical Cross-Entropy, Binary Cross-Entropy, MSE.

## 📦 Installation

You can install Ronet via pip (coming soon):

```bash
pip install ronet