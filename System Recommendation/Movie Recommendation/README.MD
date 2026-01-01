# 🎬 Movie RECOMMENDATION SYSTEM

## PURPOSE
This project aims to build recommendation system that can provide users with movie suggestion based on their preference. The system using a **Knowledge Graph Convolutional Network (KGCN)** approach to model complex interactions between users, movies, and their attributes. This project developed using python and runs through Google Colab.

## OVERVIEW
The development process is split into several key:
1. **Data Loading & Preprocessing:** 
    * loading raw data from `movies.  csv`, `users.csv`, and `ratings.csv`.
    * Merge datasets to create a unified view of user interaction
    * Encoding categorical features (Gender, Genres) into numerical formats

2. **Knowledge Graph Constructions:**
    * Transform data into **Heterogeneous Graph**
    * Defining nodes (`User`, `Movie`, `Gender`, `Age`, `Occupation`, `Genre`) and edges (create relationship `user rates movies`, `user, has gender`,`user has age`, `user has occupation`, `Movie has genre`)

3. **Data Splitting**
    * Split interactions edges into **Training (80%)**, **Validation (10%)**, and **Testing(10%)** to ensure robust model evaluation.

4. **Modelling & Evaluation:**
    * Building a graph neural network model using **GraphSAGE**
    * Training model to perform **Link Prediction**
    * Evaluating performance using metrics **AUC**, **Precision@K**, **NDCG@K**, **Precision@K**, **Recall@K**

## 🚨 Key Highlight
This project implements advanced techniques to optimize performace and accuracy

* **❗️Importance Sampling**
    Instead using uniform sampling, this project employs **Importance Sampling** during the training phase.
    * **Relevance:** Focus calculations on significant relationships, making recommendations more personalized and relevant.
    * **Efficiency** Reduces computational time and memory usage by avoiding unnecessary calculations on uninformative edges.

* **Heterogenous Graph Modelling**
    this model treats Users, Movies, and Attributes as different nodes types, allowing network to learn rich semantic relationships

## PROJECT STRUCTURE
| Description | Link |
| :--- | :--- |
| Setup data, preprocessing, and create graph | [Exploration and Preprocessing](https://github.com/RizkiYanuar-Tech/Project/blob/master/System%20Recommendation/Movie%20Recommendation/AUC%2C_Recall_K%2C_Precision_K.ipynb) |
| Building Model & Evaluation | [Building Knowledge Graph Convolutional Network (KGCN) and Evaluation Model](https://github.com/RizkiYanuar-Tech/Project/blob/master/System%20Recommendation/Movie%20Recommendation/AUC%2C_Recall_K%2C_Precision_K.ipynb) |

## 🧾 Dataset
The dataset used sourced from [Movielens-1M](https://grouplens.org/datasets/movielens/) contains three main file:
1. `movies.csv`:
- MovieID: ID Movie
- Title: Title Movies
- Genres: Genres found in the movie
    * Action
	* Adventure
	* Animation
	* Children's
	* Comedy
	* Crime
	* Documentary
	* Drama
	* Fantasy
	* Film-Noir
	* Horror
	* Musical
	* Mystery
	* Romance
	* Sci-Fi
	* Thriller
	* War
	* Western

2. `users.csv` :
- UserID: ID User
- Gender: Gender user by a "M" for male and "F" for female
- Age: User Age by range
    * 1:  "Under 18"
	* 18:  "18-24"
	* 25:  "25-34"
	* 35:  "35-44"
	* 45:  "45-49"
	* 50:  "50-55"
	* 56:  "56+"

- Occupation: User Occupation represented by number
    *  0:  "other" or not specified
	*  1:  "academic/educator"
	*  2:  "artist"
	*  3:  "clerical/admin"
	*  4:  "college/grad student"
	*  5:  "customer service"
	*  6:  "doctor/health care"
	*  7:  "executive/managerial"
	*  8:  "farmer"
	*  9:  "homemaker"
	* 10:  "K-12 student"
	* 11:  "lawyer"
	* 12:  "programmer"
	* 13:  "retired"
	* 14:  "sales/marketing"
	* 15:  "scientist"
	* 16:  "self-employed"
	* 17:  "technician/engineer"
	* 18:  "tradesman/craftsman"
	* 19:  "unemployed"
	* 20:  "writer"

3. `ratings.csv`:
- UserIDs range between 1 and 6040 
- MovieIDs range between 1 and 3952
- Ratings are made on a 5-star scale
- Timestamp is represented in seconds since the epoch as returned by ti(2)
- Each user has at least 20 ratings

## 📚 Library
* **Python**
* **Pandas**
* **Matplotlib**
* **PyTorch (Torch):** Framework Deep Learning (version 2.0.0).
* **PyTorch Geometric (PyG):** Library khusus untuk Graph Neural Network (version 2.7.0).
* **Torch Scatter:** Operasi sparse updates untuk Graph (version 2.1.2).
* **Scikit-learn:** Preprocessing data dan metrik evaluasi standar.
* **Torchmetrics:** Koleksi metrik evaluasi khusus untuk PyTorch (version 1.7.3).

## Installation
### installation for colab/standard setup
```bash
conda install pytorch==2.0.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install git+https://github.com/pyg-team/pytorch_geometric.git
conda install pandas matplotlib scikit-learn torchmetrics -c conda-forge
```

### Installation for conda env
1. create and activate a new environment
```bash
conda create -n nama_environment python=3.10
conda activate nama_environment
```

2. Install library on requirements.txt
```bash
pip install -r requirements.txt
```

# 📈 Model Performance
The model with importance sampling demonstrates strong performance in predicting user preferences.
*   ROC AUC: [0.8798] - Good capability to in distinguishing liked and disliked movies.
*   Recall@K: [1.0000] - Retrieves most relevant movies from the dataset
*   Precision@K: [0.4988] - ensures top recommendation are relevant for user
*   NDCG@K: [0.9719] - Evaluates ranking quality, higher scores mean the movie appear at the very top recommendation list