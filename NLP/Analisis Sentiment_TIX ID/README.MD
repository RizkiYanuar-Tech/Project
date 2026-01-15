# Sentiment Analysis: TIX ID Mobile App Reviews
This project focuses on analyzing user sentiment towards the TIX ID application (a popular movie ticketing platform in Indonesia) by processing user reviews. It utilizes Natural Language Processing (NLP) techniques and compares tree-based machine learning algorithms (Decision Tree and Random Forest) to classify reviews into positive or negative sentiments.

# 🎯 Purpose
The main objectives of this project are:

- To automate the process of classifying user reviews from the Google Play Store.

- To understand the general sentiment (Positive/Negative) of TIX ID users.

- To compare the performance of Decision Tree and Random Forest algorithms in text classification tasks.

# 📖 Overview
The project follows a standard NLP pipeline:
1. Data Collection: Scraping/gathering reviews from the TIX ID application page on Google Play Store.

2. Preprocessing: Cleaning the text data (Case folding, Tokenization, Stopword removal, Stemming using Sastrawi for Indonesian text).

3. Feature Extraction: Converting text into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).

4. Modeling: Training and comparing two machine learning models:

- Decision Tree: A simple, interpretable model that splits data based on feature values.

- Random Forest: An ensemble learning method that constructs a multitude of decision trees to improve accuracy and control over-fitting.

5. Evaluation: Measuring model performance using Accuracy, Precision, Recall, and F1-Score.

# 📂 Project Structure
```bash
├── NLP/
│   └── Analisis Sentiment_TIX ID/
|       ├── README.MD # Project Documentation
│       ├── Scraping_Data_TIXID_Muhammad_Rizki_Yanuar.ipynb
│       ├── Submission_Analisis_Sentiment_Muhammad_Rizki_Yanuar.ipynb
│       ├── TIX_ID REVIEWS.csv       # Dataset
│       └── requirements.txt         # List of dependencies
```

# 💾 Dataset
Source: Google Play Store Reviews (TIX ID App).
Attributes:
- UserName: Name of the reviewer.
- Content: The text review provided by the user.
- Score: Rating given (1-5 stars).
- Label: Sentiment label (derived from score, e.g., 5 -> Positive, 1 -> Negative).

# 🛠 Libraries
This project relies on the following Python libraries:
- Pandas
- NumPy
- Matplotlib / Seaborn
- NLTK / Sastrawi
- Scikit-learn
- WordCloud

# ⚙️ Installation
You can set up the environment using Conda or a standard Python Virtual Environment.
Option 1: Using Conda
```bash
# 1. Create a new environment
conda create -n sentiment-tixid python=3.9

# 2. Activate the environment
conda activate sentiment-tixid

# 3. Install required libraries
conda install pandas scikit-learn matplotlib seaborn nltk
pip install Sastrawi wordcloud
```
Option 2: Using Standard Environment (venv)
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate the environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install pandas scikit-learn matplotlib seaborn nltk Sastrawi wordcloud
```

# 🚀 How to Run
1. Clone this repository or download the project folder.
2. Ensure you have installed all dependencies.
3. Open the Jupyter Notebook:
```bash
jupyter notebook "Scraping_Data_TIXID_Muhammad_Rizki_Yanuar.ipynb"
```
4. Run the cells sequentially to scrape data from google play
5. Open the Jupyter Notebook:
```bash
jupyter notebook "Submission_Analisis_Sentiment_Muhammad_Rizki_Yanuar.ipynb"
```
6. Run the cells sequentially to perform data loading, preprocessing, training, and evaluation.

# 📊 Model Performance
The project compares the performance of two models. Below are the results obtained from the testing dataset:
1. splitting 80:20


2. splitting 70:30
