# 🌸 Iris Flower Classification using Neural Networks (PyTorch)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Dataset](https://img.shields.io/badge/Dataset-Iris-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Overview
This project implements a **Neural Network using PyTorch** to classify Iris flowers into three species:

- 🌿 Setosa  
- 🌱 Versicolor  
- 🌸 Virginica  

It demonstrates a complete **Deep Learning pipeline**:
> Data Preprocessing → Model Building → Training → Evaluation

---

## 🧠 Model Architecture

The neural network is a simple feedforward model:

- **Input Layer:** 4 features  
- **Hidden Layer 1:** 8 neurons (ReLU)  
- **Hidden Layer 2:** 9 neurons (ReLU)  
- **Output Layer:** 3 neurons  

```python
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
```

## 📊 Dataset Information
- 📁 Dataset: Iris Dataset
- 📈 Total Samples: 150
- 🔢 Features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width

## ⚙️ Tech Stack
- 🐍 Python
- 🔥 PyTorch
- 📊 Pandas
- 📈 Matplotlib
- 🤖 Scikit-learn

## 🚀 Workflow
- Load dataset
- Convert categorical labels into numeric values
- Split data into training and testing sets
- Convert data into PyTorch tensors
- Build Neural Network model
- Train model using:
  - Loss Function: CrossEntropyLoss
  - Optimizer: Adam
  - Evaluate model performance

## 🏋️ Training Details
- Epochs: 200
- Learning Rate: 0.01
- Optimizer: Adam

### Example output:
```bash
Epoch 0, Loss: 1.1634
Epoch 50, Loss: 0.2951
Epoch 100, Loss: 0.0732
```

## 📈 Results
- Model achieves high classification accuracy
- Loss decreases steadily over training
- Successfully predicts all three classes
