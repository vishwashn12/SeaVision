# Airbus Ship Detection Using Machine Learning and Deep Learning Approaches

Detecting ships in satellite imagery is critical for maritime surveillance, environmental monitoring, and navigation safety. This project explores three approaches for binary ship detection using the Airbus Ship Detection dataset:

- **Traditional Machine Learning (handcrafted features)**  
- **Simple Convolutional Neural Network (CNN)**  
- **Transfer Learning CNN (DenseNet169)**

---

## 📘 Table of Contents
1. [Folder Structure](#folder-structure)  
2. [Setup Instructions](#setup-instructions)  
3. [Dataset](#dataset)  
4. [Model Descriptions](#model-descriptions)  
5. [Running the Notebooks](#running-the-notebooks)  
6. [GPU Notes](#gpu-notes)  
7. [Optional GPU Setup](#optional-gpu-setup)  
8. [Reference Paper](#reference-paper)  
9. [Authors](#-authors)

---

## 📁 Folder Structure
```
Airbus_Ship_Detection/
│
├── models/
│   ├── Traditional.ipynb
│   ├── Simple_CNN.ipynb
│   └── Transfer_Learning_CNN.ipynb
│
├── data/
│   ├── train_v2/   # Place images here after download
│   └── train_ship_segmentations_v2.csv
│
├── Airbus_Ship_Detection_Report.pdf
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Install Python 3.11.13
Ensure you have Python 3.11.13 installed.

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 3. Clone this repository
```bash
git clone https://github.com/vishwashn12/Airbus_Ship_Detection.git
cd Airbus_Ship_Detection
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Download dataset from Kaggle
[Airbus Ship Detection Dataset](https://www.kaggle.com/competitions/airbus-ship-detection)

### 6. Place dataset files in `data/` folder as shown in folder structure.

### 7. **Update dataset paths in notebook files if necessary.**  
    Each notebook may reference dataset paths directly (e.g., `'../data/train_v2/'` or `'data/train_v2/'`).
    Ensure that these paths point to your local dataset directory.

---

## 🛰️ Dataset
| File / Folder | Description |
|----------------|--------------|
| `train_v2/` | Contains 768×768 RGB satellite images |
| `train_ship_segmentations_v2.csv` | CSV with image IDs and ship masks (EncodedPixels) |

> **Note:** Dataset size ~30 GB — cannot be pushed to GitHub.

---

## 🧠 Model Descriptions
| Model | Input Size | Training Data | Key Features / Notes | Validation Accuracy |
|--------|-------------|----------------|------------------------|---------------------|
| **Traditional ML** | 256×256 | 15,000 images (natural imbalance) | Hu Moments, Haralick features, Color Histograms; LDA, KNN, Naive Bayes, SVM, Random Forest (n_estimators=140, max_depth=30) | **92.48%** |
| **Simple CNN (Custom)** | 128×128 | 70,000 images (balanced) | 3 Conv blocks + fully connected layers; Data augmentation (rotation, flips); Adam optimizer (1e-4), BCE loss | **93.97%** |
| **Transfer Learning CNN** | 256×256 | 10,000 images (original imbalance) | DenseNet169 pretrained; Classification head: BatchNorm → Dropout(0.5) → GlobalMaxPooling → Dense(1664, ReLU) → Dense(128, ReLU) → Dense(1, Sigmoid); Adam optimizer (1e-4), EarlyStopping, Threshold 0.6 | **93.9%** |

---

## ▶️ Running the Notebooks
| Notebook | Description |
|-----------|--------------|
| `Traditional.ipynb` | Trains and evaluates handcrafted feature ML models (CPU-only) |
| `Simple_CNN.ipynb` | Trains a custom CNN using PyTorch; supports GPU or CPU |
| `Transfer_Learning_CNN.ipynb` | Fine-tunes DenseNet169 with TensorFlow; supports GPU or CPU |

### Run using Jupyter:
```bash
jupyter notebook
```
Open the desired notebook and run the cells sequentially.

> Ensure the dataset is in the `data/` folder. Trained models will be saved automatically within notebook cells if enabled.

---

## ⚡ GPU Notes
- **TensorFlow**: Automatically uses GPU if available; otherwise falls back to CPU.
- **PyTorch**: Device is auto-selected (`cuda` if available, else `cpu`).
- GPU significantly speeds up training for the Simple CNN and Transfer Learning models.
- CPU execution is slower but fully functional.

---

## 💻 Optional GPU Setup

### TensorFlow GPU
```bash
# Uninstall CPU version first
pip uninstall tensorflow -y
pip install tensorflow-gpu==2.15.0
```

### PyTorch GPU
```bash
# Example for CUDA 12.1
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

The **Simple_CNN** notebook automatically detects GPU:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 📑 Reference Paper
- **[Airbus Ship Detection - CS229 Project Report](https://cs229.stanford.edu/proj2018/report/58.pdf)**
- **[Dataset (Airbus Ship Detection Challenge)](https://www.kaggle.com/competitions/airbus-ship-detection)**

## 🖋️ Authors
**Vishwas H N**<br>
**Vikas S**