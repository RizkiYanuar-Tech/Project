# 🗑️ Klasifikasi Gambar: Garbage Dataset
Project ini merupakan model deep learning untuk mengklasifikasikan jenis sampah ke dalam 10 kategori berbeda menggunakan teknik transfer learning dengan arsitektur MobileNetV2.

## 🎯 PURPOSE
Tujuan project ini adalah untuk mengembangkan model klasifikasi gambar yang akurat untuk mengenali berbagai jenis sampah.

## 📖 Overview
Project ini menggunakan Convolutional Neural Network (CNN) + transfer learning
-   Base Model: MobileNetV2 (Pre-trained on ImageNet).
-   Feature Extraction: Layer pada MobileNetV2 dibekukan kecuali 10 layer terakhir untuk dilakukan fine-tuning
-   Custom Layers: Menambahkan layer konvolusi, GlobalAveragePooling, dan Dense Layer untuk klasifikasi spesifik.
-   Input Shape: Merubah ukuran citra (resize ) untuk menyesuaikan input model

## 📂 Project Structure
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

## 🧾 Dataset
Dataset yang digunakan berasal dari Kaggle: Garbage Classification V2
*   Sumber: [Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
*   Jumlah Kategori: 10 Kategori
*   Daftar Kelas:

    1.  Battery

    2.  Biological

    3.  Cardboard

    4.  Clothes

    5.  Glass

    6.  Metal

    7.  Paper

    8.  Plastic

    9.  Shoes

    10. Trash

## 📚 Library
Library yang digunakan dalam proyek:
*   TensorFlow * Keras
*   Numpy
*   Matplotlib & Seaborn
*   Pandas
*   Scikit-learn
*   OpenCV & PIL

## Installation
## Menggunakan Conda
```bash
# 1. Buat environment baru
conda create -n garbage-class python=3.9

# 2. Aktifkan environment
conda activate garbage-class

# 3. Install library yang dibutuhkan
conda install tensorflow pandas matplotlib seaborn scikit-learn opencv

pip install scikit-image
```

## Menggunakan starndar Environment (venv)
```bash
# 1. Buat virtual environment
python -m venv venv

# 2. Aktifkan environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install tensorflow pandas matplotlib seaborn scikit-learn opencv-python scikit-image
```
# 🚀 How to Run
1. Pastikan dataset sudah diunduh. Jika menggunakan Kaggle API (seperti di notebook), pastikan file kaggle.json sudah tersedia.

2. Buka Jupyter Notebook atau Google Colab.

3. Jalankan file Klasifikasi_Gambar.ipynb.

4. Notebook akan melakukan ekstraksi data, preprocessing, pelatihan model, dan evaluasi.

## 📊 Model Performance
Berdasarkan hasil pelatihan terakhir (Epoch 10), model mencapai performa berikut:
*   Training Accuracy: 94%

*   Validation Accuracy: 92%

*   Test Accuracy: 92%

*   Optimizer: Adam (Learning Rate: 5e-4)