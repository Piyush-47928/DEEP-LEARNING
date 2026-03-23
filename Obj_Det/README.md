# 🧠 AI Object Detection System (YOLOv8 + Google Colab)

An AI-powered object detection system that analyzes images and outputs structured results including object counts, bounding boxes, and confidence scores.

---

## 🚀 Features

- Detects multiple objects in a single image
- Uses YOLOv8 (Ultralytics) for fast & accurate detection
- Structured output format:
  - Total objects
  - Object-wise counts
  - Bounding boxes (normalized)
  - Confidence scores
- Unique ID for each object
- Annotated image visualization
- JSON export support
- Runs on Google Colab (no setup required)

---

## 📦 Tech Stack

- Python 3
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Matplotlib
- Google Colab

---

## 🖼️ Input
```bash
Upload any image file (`.jpg`, `.png`, etc.) via Colab interface.
```

---

## 📤 Output Format
```bash
Example:
Total Objects Detected: 14

- Objects Detected:
potted plant: 3
cell phone: 1
vase: 1
laptop: 1
remote: 1
couch: 1
bowl: 1
cup: 1
tv: 1
dining table: 1
book: 2
```

---

## ⚙️ How to Use

### 1. Open Google Colab
https://colab.research.google.com/

### 2. Paste the Code
Copy the provided Python script into a Colab cell.

### 3. Run the Cell
Dependencies will install automatically.

### 4. Upload Image
Use the upload prompt.

### 5. View Results
- Structured console output
- Annotated image display
- JSON file generated

---

## 📊 Detection Details

- Bounding boxes are normalized (0 to 1)
- Confidence score represents prediction certainty
- Labels follow COCO dataset classes

---

## ⚠️ Limitations

- Limited to YOLOv8 pretrained classes (~80 categories)
- Rare/unusual objects may not be detected
- Scene understanding is basic

---

## 🔧 Customization

### Use a larger model (better accuracy)
```python
model = YOLO("yolov8m.pt")
```

## 🧪 Future Improvements
- Custom dataset training
- Video detection
- API deployment (Flask/FastAPI)
- Advanced scene understanding
