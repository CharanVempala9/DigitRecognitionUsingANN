# 🔢 Handwritten Digit Recognition using Multilayer Perceptron (MLP)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview

This project implements a **Multilayer Perceptron (MLP)** neural network using **TensorFlow/Keras** to classify handwritten digits from the **MNIST dataset**. The model is trained to recognize digits (0–9) by analyzing grayscale images and predicting the correct digit using a feedforward architecture.

---

## 🚀 Demo

👉 **Interactive Demo:** [Coming Soon – TF.js Version]  
👉 **Model File:** [`digits_recognition_mlp.h5`](https://trekhleb.dev/machine-learning-experiments/#/experiments/DigitsRecognitionMLP
)   
---

## 🎯 Features

- Uses **MLP Neural Network** with ReLU & Softmax activations
- Trained on **60,000+ MNIST images**, tested on 10,000
- Achieves **~97% training** and **~96% test accuracy**
- Includes:
  - 🔍 Confusion Matrix visualization
  - 📈 Loss & Accuracy tracking
  - 🎯 Model Saving & TensorBoard Logs
- Ready for **TensorFlow.js** conversion and web deployment

---

## 🧠 Technologies Used

| Tool/Library     | Purpose                                |
|------------------|----------------------------------------|
| TensorFlow/Keras | Building and training neural networks  |
| NumPy            | Numerical computations                 |
| Matplotlib       | Data visualization                     |
| Seaborn          | Confusion matrix heatmap               |
| TensorBoard      | Training process visualization         |

---

## 📂 Dataset – MNIST

| Split     | Images Count | Image Size |
|-----------|--------------|------------|
| Training  | 60,000       | 28x28      |
| Testing   | 10,000       | 28x28      |

Each image represents a **handwritten digit** from 0 to 9 in grayscale.

---

## 🏗️ Model Architecture

```text
Input Layer      → Flatten(28x28 = 784)
Hidden Layer 1   → Dense(128, ReLU)
Hidden Layer 2   → Dense(128, ReLU)
Output Layer     → Dense(10, Softmax)
