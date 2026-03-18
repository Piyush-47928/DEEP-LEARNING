# 🧠 Crowd Surveillance System  
### *(Supervised LLE + Supervised UMAP + Siamese Network)*

---

## 📌 Overview  
This repository showcases my **Crowd Surveillance Project**, designed to analyze crowd scenes and detect anomalies using a hybrid deep learning architecture.

The system leverages:
- **Supervised LLE (Locally Linear Embedding)** for local structure preservation  
- **Supervised UMAP (Uniform Manifold Approximation and Projection)** for global structure learning  
- **Siamese Neural Network** for similarity-based anomaly detection  

The goal is to build an efficient and scalable model for **real-time surveillance applications**.

---

## 🚀 Features  
- 🔍 Crowd anomaly detection  
- 🧠 Hybrid dimensionality reduction (LLE + UMAP)  
- 🔗 Siamese architecture for similarity learning  
- ⚡ Optimized for training in limited compute environments (e.g., hackathons)  
- 📊 Structured training & evaluation pipeline  

---

## 🏗️ Architecture  

```bash
          Input Image
             ↓
CNN Feature Extractor (Embedding: 512D)
             ↓
  Supervised LLE (512 → 128)
             ↓
Supervised UMAP (128 → Lower Dim Space)
             ↓
Siamese Network (Similarity Learning)
             ↓
    Anomaly Detection Output
```


---

## 📂 Dataset  

- The dataset used in this project is sourced from **Kaggle**.  
- It consists of crowd surveillance images used for training and evaluation.

> ⚠️ Note: Due to dataset licensing, the images are not included in this repository.  
> Please download the dataset directly from Kaggle and place it in the appropriate directory.

---

## 📁 Project Structure  

```bash
Crowd-Surveillance/
│── data/ # Dataset (not included)
│── models/ # Model architecture files 
│── LLE (1st model created)
│── UMAP+Supervised_LLE (Main file)
│── README.md
```


---

## ⚙️ Installation  

### Clone the repository
```bash
git clone https://github.com/Piyush-47928/Crowd_Survill.git
```
### Navigate to project folder
```bash
cd Crowd_Survill
```
### Install dependencies
```bash
See the **requirements.md** file and then install your virtual environment folder
```
## 📊 Results

- Improved feature representation using LLE + UMAP pipeline
- Effective similarity learning via Siamese Network
- Capable of distinguishing normal vs anomalous crowd behavior

## 🧪 Future Improvements

- Real-time video stream integration
- Attention-based feature extraction
- Transformer-based embeddings
- Deployment on edge devices (CCTV systems)

