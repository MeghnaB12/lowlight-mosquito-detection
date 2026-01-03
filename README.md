# Mosquito Species Detection (YOLOv8)

This project contains the complete code for training a state-of-the-art Object Detection model to classify and localize different mosquito species. The solution utilizes YOLOv8 (You Only Look Once), a cutting-edge real-time object detection architecture, to achieve high accuracy on high-resolution images.

# 🚀 Key Features

The pipeline is engineered for precision and automation within a Kaggle environment, leveraging the Ultralytics framework for streamlined training and inference.

* SOTA Object Detection: Utilizes YOLOv8m (Medium), offering an optimal balance between inference speed and detection accuracy for small objects like insects.

* High-Resolution Training: Configured to train on 1024x1024 image sizes, ensuring the model captures fine-grained features essential for distinguishing between similar mosquito species.

* Automated Data Pipeline: Includes scripts for dynamic data splitting (90/10 Train/Val), directory restructuring, and YAML configuration generation.

* Robust Inference Logic: Features a custom post-processing script that parses raw YOLO detections, filters by confidence, handles missing predictions, and formats the output for submission.



# 📈 Methodology
The core of this solution is a supervised object detection pipeline that draws bounding boxes around mosquitoes and classifies them into one of six distinct species.

## 1. Data Preparation

To adapt the raw Kaggle dataset for YOLO format, the following steps are automated:

* Splitting: The dataset is shuffled and split into Training (90%) and Validation (10%) sets.

* Restructuring: Images and labels are copied into a standard dataset/images and dataset/labels hierarchy required by Ultralytics.

* Configuration: A dataset.yaml file is dynamically generated to define paths and map class indices to names: ['aegypti', 'albopictus', 'anopheles', 'culex', 'culiseta', 'japonicus/koreicus'].

### 2. Model Architecture (YOLOv8)

The model is built on the Ultralytics YOLOv8 framework:

* Backbone: CSPDarknet53-based feature extractor.

* Head: Anchor-free detection head that predicts class probabilities and bounding box coordinates simultaneously.

* Training Configuration:

Epochs: 30
Image Size: 1024 px (to detect small details)
Batch Size: 16
Optimizer: Auto (SGD/AdamW)

### 3. Inference & Submission

* Prediction: The trained model runs inference on the test set with a confidence threshold of 0.30 and IoU threshold of 0.50.

* Post-Processing: Raw .txt outputs are parsed to extract the highest-confidence detection for each image.

* Fallback Mechanism: If no object is detected, the script assigns a default class and bounding box to ensure the submission file remains valid.

# 🛠️ Tech Stack

* Core: Python 3
* Deep Learning: PyTorch, Ultralytics YOLOv8
* Data Handling: Pandas, NumPy, Shutil

# 🏃 Running the Project

### 1. Dependencies

This script is designed to run in a Kaggle Notebook with GPU acceleration.

```
pip install ultralytics pandas numpy
```

### 2. Dataset

This model was trained on a mosquito classification dataset as part of a university challenge. The data consists of high-resolution images of mosquitoes and their corresponding bounding box annotations (YOLO format). The classes include Aedes aegypti, Aedes albopictus, Anopheles, Culex, Culiseta, and Japonicus/Koreicus.

Due to privacy and access restrictions, the dataset is not publicly available and is not included in this repository. Therefore, the script cannot be run out-of-the-box without downloading the specific competition data separately and placing it in the correct directory structure.

### 3. Notebook Review

The provided code serves as an end-to-end pipeline:

* Data Prep: Moves files and creates the YAML config.

* Training: Fine-tunes yolov8m.pt on the custom dataset.

* Validation: Reports mAP50 and mAP50-95 metrics.
