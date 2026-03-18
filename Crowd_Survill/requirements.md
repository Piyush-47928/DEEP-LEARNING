### Core Libraries

numpy==1.24.4
scipy==1.10.1
pandas==2.0.3

---

### Deep Learning (PyTorch)

torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2

---

### Machine Learning

scikit-learn==1.3.2
umap-learn==0.5.5

---

### Image Processing

opencv-python==4.8.1.78
Pillow==10.0.1

---

### Visualization

matplotlib==3.7.2
seaborn==0.12.2

---

### Progress & Utilities

tqdm==4.66.1
joblib==1.3.2

---

### Optional (Performance / Logging)

tensorboard==2.14.1

---

### ⚠️ Important Notes (Read This)

- If you're using Google Colab, you can remove version pins (or loosen them).

- If CUDA issues come, install PyTorch separately from official site:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- **umap-learn** is essential for your pipeline (don’t skip it).
```
- umap-learn is essential for your pipeline (don’t skip it).
