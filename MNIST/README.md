# 🧠 Convolutional Neural Network (CNN) on MNIST Dataset

This repository showcases the implementation of a **Convolutional Neural Network (CNN)** for handwritten digit classification using the **MNIST dataset**. The project demonstrates how deep learning models can effectively recognize and classify images.

---

## 📌 Project Overview

The goal of this project is to build and train a CNN model that can accurately classify handwritten digits (0–9) from grayscale images.

The MNIST dataset is widely used as a benchmark in computer vision and consists of **28x28 pixel images** of handwritten digits.

---

## 🎯 Objectives

- Understand the working of **Convolutional Neural Networks**
- Implement CNN using **TensorFlow / Keras**
- Train a model on image data
- Evaluate model performance
- Visualize predictions

---

## 🛠️ Technologies Used

- Python 3
- NumPy
- Matplotlib
- TensorFlow / Keras
- Jupyter Notebook / VS Code

---

## 📂 Dataset

The dataset used in this project is the **MNIST handwritten digits dataset**, which contains:

- 60,000 training images
- 10,000 testing images
- 10 classes (digits 0–9)

Each image is:
- Grayscale
- 28 × 28 pixels

---

## 🧠 CNN Architecture

The model consists of the following layers:

```bash
Input Layer (28x28x1)
↓
Convolution Layer (32 filters, 3x3) + ReLU
↓
Max Pooling (2x2)
↓
Convolution Layer (64 filters, 3x3) + ReLU
↓
Max Pooling (2x2)
↓
Flatten Layer
↓
Dense Layer (128 neurons) + ReLU
↓
Output Layer (10 neurons) + Softmax
```

---

## ⚙️ How to Run the Project
### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Piyush-47928/CNN
cd CNN-MNIST
```

### 2️⃣ Install Dependencies

```bash
pip install numpy matplotlib tensorflow
```

### 3️⃣ Run the Code

```bash
jupyter notebook CNN-MNIST.ipynb
```

---

## 📊 Model Performance

```bash
Example :
Epoch 20/20
Train Loss: 0.0087 | Train Acc: 99.72%
Test Loss: 0.0473 | Test Acc: 98.97%
```
- Final Accuracy: ~98–99%
- Loss significantly reduced over epochs

---

## 📈 Results & Visualization

- Model successfully classifies handwritten digits
- Predictions can be visualized using Matplotlib
- Confusion matrix can be used for deeper evaluation

