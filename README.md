# Image Classification Pipeline

This project provides an end-to-end pipeline for image classification and object detection using PyTorch and torchvision. It includes data preprocessing, model training, and inference for both classification and detection tasks.

## Features
- Object detection using YOLO
- Image classification using ResNet
- Data preprocessing utilities
- COCO-format dataset validation script

## Directory Structure
```
.
├── main.py                        # Main pipeline script
├── train_classification.py        # Training script for classification
├── classification/                # Classification models and code
│   ├── resnet.py
│   └── __init__.py
├── detection/                     # Detection models and code
│   ├── yolo.py
│   └── __init__.py
├── utils/                         # Preprocessing utilities
│   ├── image_classification_preprocess.py
│   ├── video_preprocess.py
│   └── types.py
└── data/                          # Data and datasets
```

## Setup
1. **Clone the repository**
   ```bash
   git clone <this-repo-url>
   cd image-classification-pipeline
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision
   pip install -r requirements.txt  # If available
   ```

3. **Download the dataset**
   Download the MobilCoco dataset from Kaggle:
   ```bash
   mkdir -p data
   curl -L -o ./data/archive.zip https://www.kaggle.com/api/v1/datasets/download/gunawan26/labelled-indonesian-car-and-plate-number
   # Unzip the dataset
   unzip ./data/archive.zip -d ./data/
   ```

## Usage


### 1. Train the Classifier
```bash
python train_classification.py
```

### 2. Run the Main Pipeline
```bash
python main.py
```

## Notes
- The main pipeline expects an input video at `./data/input.mp4`. Which you can get from https://intip.in/QNpw
- Model weights will be saved as `classifier_best_weights.pth` after training.
- Adjust paths and parameters as needed for your use case.

## License
MIT License
