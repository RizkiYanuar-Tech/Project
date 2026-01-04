# 🗑️ Klasifikasi Gambar: Garbage Dataset
This project is a deep learning model for classifying waste types into 10 different categories using transfer learning techniques with the MobileNetV2 architecture.

## 🎯 PURPOSE
The aim of this project is to develop an accurate image classification model to recognise various types of waste.

## 📖 Overview
This project uses Convolutional Neural Network (CNN) + transfer learning
-   Base Model: MobileNetV2 (Pre-trained on ImageNet).
- Feature Extraction: Layers in MobileNetV2 are frozen except for the last 10 layers for fine-tuning
- Custom Layers: Adding convolution layers, GlobalAveragePooling, and Dense Layers for specific classification.
- Input Shape: Resizing images to fit the model input

## 📂 Project Structure
```bash
├── modeltfjs/
│   └── model.json
│── saved_model/ 
    └── variables/
        │── variables.data-0000-of-00001
        └── variables.index
│── tflite/
    │──labels.txt
    └── model1.tflite
│── Klasifikasi_Gambar.ipynb
│── README.MD
└── requirements.txt
```

## 🧾 Dataset
The dataset used is from Kaggle: Garbage Classification V2
*   Source: [Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
*   Number of Categories: 10 Kategori
*   Class List:

    *  Battery

    *  Biological

    *  Cardboard

    *  Clothes

    *  Glass

    *  Metal

    *  Paper

    *  Plastic

    *  Shoes

    * Trash

## 📚 Library
Libraries used in the project:
*   TensorFlow 
*   Keras
*   Numpy
*   Matplotlib & Seaborn
*   Pandas
*   Scikit-learn
*   OpenCV & PIL

## Installation
## Conda environment
```bash
# 1. create new environment
conda create -n garbage-class python=3.9

# 2. activate environment
conda activate garbage-class

# 3. Install library
conda install tensorflow pandas matplotlib seaborn scikit-learn opencv

pip install scikit-image
```

##  standard Environment (venv)
```bash
# 1. create virtual environment
python -m venv venv

# 2. activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install tensorflow pandas matplotlib seaborn scikit-learn opencv-python scikit-image
```
# 🚀 How to Run
1. Ensure that the dataset has been downloaded. If using the Kaggle API (as in the notebook), ensure that the kaggle.json file is available.

2. Open Jupyter Notebook or Google Colab.

3. Run the file Klasifikasi_Gambar.ipynb.

4. The notebook will perform data extraction, preprocessing, model training, and evaluation.

## 📊 Model Performance
Based on the results of the last training (Epoch 10), the model achieved the following performance:
*   Training Accuracy: 94%

*   Validation Accuracy: 92%

*   Test Accuracy: 92%

*   Optimizer: Adam (Learning Rate: 5e-4)
