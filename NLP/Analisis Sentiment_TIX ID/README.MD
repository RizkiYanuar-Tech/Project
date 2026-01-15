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
│       ├── data/
│       │   ├── raw_reviews.csv      # Original scraped data
│       │   └── clean_data.csv       # Preprocessed dataset
│       ├── notebook/
│       │   └── Sentiment_Analysis.ipynb  # Main Jupyter Notebook
│       ├── models/
│       │   ├── dt_model.pkl         # Saved Decision Tree model (optional)
│       │   └── rf_model.pkl         # Saved Random Forest model (optional)
│       ├── README.md                # Project Documentation
│       └── requirements.txt         # List of dependencies
```
