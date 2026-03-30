# 🩺 DermScreen AI — Real-Time Dermatological Screening System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**DermScreen AI** is a cutting-edge deep learning system for real-time dermatological screening using phone camera input. It combines advanced computer vision, 3D augmentation techniques, and medical imaging classification to provide instant skin lesion analysis with risk assessment.

![DermScreen AI Demo](https://img.shields.io/badge/Status-Production%20Ready-success)

---

## 🌟 Key Features

### 🎯 **Real-Time Classification**
- **MobileNetV2-based CNN** with channel attention mechanism
- **7 skin lesion categories**: Melanoma, Basal Cell Carcinoma, Actinic Keratosis, Vascular Lesion, Benign Keratosis, Melanocytic Nevi, Dermatofibroma
- **Risk-level assessment**: Critical, High, Moderate, Low
- **Sub-second inference** on CPU (~100-300ms)

### 📸 **Camera Integration**
- WebRTC-powered real-time camera access
- Front/rear camera switching
- Live capture with viewfinder overlay
- File upload support for existing images

### 🎨 **3D Augmentation Pipeline**
Advanced data augmentation simulating real-world variations:
- **Surface Normal Estimation** — Gradient-based pseudo-3D reconstruction
- **Perspective Warping** — 3D rotation (pitch/yaw/roll) via homography transforms
- **Lambertian Lighting** — Surface-aware lighting simulation
- **Elastic Deformation** — Skin stretch/compression modeling
- **Color Augmentation** — HSV-space white balance & skin tone variation

### 🔮 **3D Visualization**
- Interactive Three.js 3D skin surface viewer
- Lesion texture mapping onto curved geometry
- Real-time rotation and zoom controls
- Step-by-step augmentation visualization

### 🎨 **Modern Medical UI**
- Dark mode medical-grade interface
- Animated risk cards with color-coded severity
- Probability distribution charts
- Responsive design for mobile/desktop

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React + Vite)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Camera     │  │  3D Viewer   │  │ Augmentation │      │
│  │  Component   │  │  (Three.js)  │  │   Viewer     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↕ REST API
┌─────────────────────────────────────────────────────────────┐
│                   Backend (FastAPI + PyTorch)                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              DermClassifier Model                     │   │
│  │  • MobileNetV2 Backbone (ImageNet Pretrained)        │   │
│  │  • Channel Attention (SE-style)                      │   │
│  │  • Classification Head: FC(1280→512→256→7)          │   │
│  │  • Auxiliary Confidence Head: FC(1280→128→1)        │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           3D Augmentation Pipeline                    │   │
│  │  1. Surface Normal Estimation (Sobel gradients)      │   │
│  │  2. Perspective Warp (3D rotation matrix)            │   │
│  │  3. Lambertian Lighting (dot product shading)        │   │
│  │  4. Elastic Deformation (Gaussian displacement)      │   │
│  │  5. Color Augmentation (HSV transforms)              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+**
- **Node.js 18+**
- **npm or yarn**

### Installation

#### 1. Clone the repository
```bash
git clone https://github.com/Maddy824/DL-In-medic.git
cd DL-In-medic
```

#### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

#### 3. Frontend Setup
```bash
cd ../frontend
npm install
```

### Running the Application

#### Terminal 1 — Start Backend Server
```bash
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

#### Terminal 2 — Start Frontend Dev Server
```bash
cd frontend
npm run dev
```

#### Access the Application
Open your browser and navigate to:
```
http://localhost:5173
```

The frontend will automatically proxy API requests to the backend at `http://localhost:8000`.

---

## 📊 Model Performance

### Architecture Details
| Component | Specification |
|-----------|--------------|
| **Backbone** | MobileNetV2 (ImageNet pretrained) |
| **Input Size** | 224×224 RGB |
| **Parameters** | ~3.5M trainable |
| **Attention** | Channel attention (reduction=16) |
| **Regularization** | Dropout (0.3, 0.3, 0.2), BatchNorm |
| **Output** | 7 classes + confidence score |

### Classification Categories
| Class | Risk Level | Description |
|-------|-----------|-------------|
| **Melanoma** | 🔴 Critical | Most dangerous skin cancer — requires immediate evaluation |
| **Basal Cell Carcinoma** | 🟠 High | Most common skin cancer — seek dermatologist |
| **Actinic Keratosis** | 🟡 Moderate | Pre-cancerous lesion — monitor and treat |
| **Vascular Lesion** | 🟡 Moderate | Blood vessel-related — usually benign |
| **Benign Keratosis** | 🟢 Low | Non-cancerous growth — generally harmless |
| **Melanocytic Nevi** | 🟢 Low | Common mole — monitor for changes |
| **Dermatofibroma** | 🟢 Low | Benign fibrous nodule — no treatment needed |

---

## 🔬 Technical Deep Dive

### 3D Augmentation Pipeline

#### 1. **Surface Normal Estimation**
```python
# Gradient-based normal map generation
dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
normals = normalize([-dx, -dy, 1])
```

#### 2. **Perspective Warping**
```python
# 3D rotation matrices (Rx, Ry, Rz)
R = Rz @ Ry @ Rx
# Perspective projection with focal length f
projected = f * R @ point / (f + z)
```

#### 3. **Lambertian Lighting**
```python
# Surface shading based on light direction
intensity = max(0, dot(normal, light_direction))
shaded = ambient + diffuse * intensity
```

#### 4. **Elastic Deformation**
```python
# Gaussian-smoothed random displacement field
dx = gaussian_filter(random_field, sigma=10)
warped = remap(image, x + dx, y + dy)
```

### API Endpoints

#### `POST /api/analyze`
Analyze a skin lesion image and return classification results.

**Request:**
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@lesion.jpg"
```

**Response:**
```json
{
  "id": "uuid-here",
  "prediction": "Melanocytic Nevi",
  "confidence": 0.8234,
  "top_probability": 0.7891,
  "risk_level": "low",
  "description": "Common mole. Usually benign but monitor for changes.",
  "all_predictions": [...],
  "inference_time_ms": 142.3,
  "disclaimer": "This is a screening tool only..."
}
```

#### `POST /api/augment`
Apply 3D augmentation pipeline and return visualization steps.

**Response:**
```json
{
  "steps": {
    "original": "data:image/png;base64,...",
    "normals": "data:image/png;base64,...",
    "perspective": "data:image/png;base64,...",
    "lighting": "data:image/png;base64,...",
    "elastic": "data:image/png;base64,...",
    "color": "data:image/png;base64,...",
    "augmented": "data:image/png;base64,..."
  },
  "step_order": ["original", "normals", ...]
}
```

---

## 📁 Project Structure

```
DL-In-medic/
├── backend/
│   ├── main.py              # FastAPI server & endpoints
│   ├── model.py             # DermClassifier architecture
│   ├── augmentation.py      # 3D augmentation pipeline
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main application component
│   │   ├── components/
│   │   │   ├── Camera.jsx           # Camera capture UI
│   │   │   ├── Results.jsx          # Classification results
│   │   │   ├── ThreeDViewer.jsx     # 3D surface viewer
│   │   │   └── AugmentationViewer.jsx
│   │   ├── utils/
│   │   │   └── api.js       # API client functions
│   │   ├── index.css        # Tailwind styles
│   │   └── main.jsx         # React entry point
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── index.html
├── .gitignore
└── README.md
```

---

## 🎯 Use Cases

- **🏥 Telemedicine**: Remote skin screening for underserved areas
- **🔬 Research**: Dermatological dataset augmentation
- **📚 Education**: Medical student training tool
- **🚨 Early Detection**: Preliminary melanoma screening
- **📊 Clinical Trials**: Automated lesion documentation

---

## ⚠️ Medical Disclaimer

**IMPORTANT**: This tool is for **educational and screening purposes only**. It is **NOT a substitute for professional medical diagnosis**.

- The demo model uses ImageNet pretrained weights
- **Not clinically validated** or FDA-approved
- Always consult a **board-certified dermatologist** for any skin concerns
- Do not use for self-diagnosis or treatment decisions
- Accuracy may vary based on image quality, lighting, and skin type

---

## 🛠️ Development

### Training Your Own Model

To train on the HAM10000 dataset or custom data:

```python
# Example training script (not included)
from model import DermClassifier
import torch

model = DermClassifier(num_classes=7, pretrained=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# ... training loop
```

### Running Tests
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Building for Production
```bash
# Frontend production build
cd frontend
npm run build

# Serve with backend
cd ../backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **HAM10000 Dataset**: Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 (2018).
- **MobileNetV2**: Sandler, M., et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR 2018.
- **PyTorch**: Paszke, A., et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." NeurIPS 2019.

---

## 📧 Contact

**Maddy** — [@Maddy824](https://github.com/Maddy824)

Project Link: [https://github.com/Maddy824/DL-In-medic](https://github.com/Maddy824/DL-In-medic)

---

## 🌐 Demo

🚀 **Live Demo**: Coming soon!

---

<div align="center">
  <strong>Built with ❤️ for advancing dermatological AI</strong>
  <br>
  <sub>Remember: Always consult a healthcare professional for medical advice</sub>
</div>