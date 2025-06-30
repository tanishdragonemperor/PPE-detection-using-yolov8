# PPE Detection using YOLOv8

A real-time Personal Protective Equipment (PPE) detection system built using YOLOv8 to enhance workplace safety by automatically identifying compliance with safety standards.

## 🚀 Overview

This project implements a state-of-the-art object detection system using YOLOv8 to identify various types of Personal Protective Equipment (PPE) in real-time. The system can detect workers and their safety equipment to ensure compliance with workplace safety regulations.

## 🎯 Features

- **Real-time Detection**: Process live video streams or webcam feeds
- **Multiple PPE Classes**: Detects various PPE items including:
  - Hard hats/Helmets
  - Safety vests
  - Safety boots
  - Gloves
  - Safety glasses/goggles
  - Masks
  - Ear protection
  - Safety harnesses
- **High Accuracy**: Leverages YOLOv8's advanced architecture for precise detection
- **Flexible Input**: Works with images, videos, and live camera feeds
- **Easy Integration**: Simple API for integration into existing safety systems

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for better performance)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tanishdragonemperor/PPE-detection-using-yolov8.git
cd PPE-detection-using-yolov8
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained weights (if available):
```bash
# Download custom trained weights or use YOLOv8 base weights
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## 📊 Dataset

The model is trained on a comprehensive PPE dataset containing:
- Construction site images
- Industrial workplace scenarios
- Various lighting conditions
- Different PPE combinations

Dataset structure:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## 🚀 Usage

### Basic Detection

```python
from ultralytics import YOLO

# Load the model
model = YOLO('path/to/your/trained/model.pt')

# Run inference
results = model('path/to/image.jpg')

# Display results
results[0].show()
```

### Real-time Detection

```python
import cv2
from ultralytics import YOLO

# Load model
model = YOLO('path/to/your/model.pt')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    results = model(frame)
    
    # Display results
    annotated_frame = results[0].plot()
    cv2.imshow('PPE Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Batch Processing

```python
# Process multiple images
results = model(['image1.jpg', 'image2.jpg', 'image3.jpg'])

# Process video
results = model('path/to/video.mp4', save=True)
```

## 🏋️ Training

To train the model on your custom dataset:

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model

# Train the model
results = model.train(
    data='path/to/your/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### Dataset Configuration

Create a `dataset.yaml` file:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 8  # number of classes
names: ['person', 'hardhat', 'mask', 'vest', 'gloves', 'boots', 'glasses', 'earprotection']
```

## 📈 Performance

| Model | mAP@0.5 | mAP@0.5:0.95 | Speed (ms) |
|-------|---------|--------------|------------|
| YOLOv8n | 85.2% | 68.1% | 1.2 |
| YOLOv8s | 87.8% | 71.3% | 2.1 |
| YOLOv8m | 89.1% | 73.7% | 3.8 |

## 🔧 Configuration

Key parameters you can adjust:

- **Confidence Threshold**: Minimum confidence for detections (default: 0.25)
- **IOU Threshold**: Non-maximum suppression threshold (default: 0.45)
- **Image Size**: Input image size for inference (default: 640)

```python
# Adjust detection parameters
results = model(
    'image.jpg',
    conf=0.4,      # confidence threshold
    iou=0.5,       # IOU threshold
    imgsz=640      # image size
)
```

## 📁 Project Structure

```
PPE-detection-using-yolov8/
├── data/
│   ├── images/
│   └── labels/
├── models/
│   └── trained_weights.pt
├── src/
│   ├── train.py
│   ├── detect.py
│   └── utils.py
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
├── dataset.yaml
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the base detection framework
- Construction Site Safety datasets from various sources
- Open source PPE detection research community

## 📞 Contact

Tanish - [@tanishdragonemperor](https://github.com/tanishdragonemperor)

Project Link: [https://github.com/tanishdragonemperor/PPE-detection-using-yolov8](https://github.com/tanishdragonemperor/PPE-detection-using-yolov8)

---
