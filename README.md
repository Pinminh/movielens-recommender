# 🎬 Movie Recommender System on MovieLens Dataset

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Surprise](https://img.shields.io/badge/Surprise-1.1.4-orange.svg)](https://surpriselib.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)

## 📖 Overview

This project is developed as part of the Computer Science curriculum at [Ho Chi Minh City University of Technology (HCMUT)](https://hcmut.edu.vn/). It serves as an introductory exploration into the field of **Recommender Systems**, implementing and comparing multiple collaborative filtering algorithms on the [MovieLens dataset](https://grouplens.org/datasets/movielens/).

The project explores several recommendation algorithms, evaluates their performance using multiple metrics, and provides a REST API for serving predictions.

## 🎯 Purpose

- **Educational**: Provide a hands-on introduction to recommender systems
- **Comparative Analysis**: Evaluate different collaborative filtering algorithms
- **Practical Application**: Build a functional movie recommendation API

## 📊 Dataset

The project uses the **MovieLens Small Dataset (ml-latest-small)**, which contains:

| Statistic | Value |
|-----------|-------|
| Ratings | 100,836 |
| Movies | 9,742 |
| Users | 610 |
| Tags | 3,683 |
| Rating Scale | 0.5 - 5.0 stars |
| Time Range | March 1996 - September 2018 |

**Data Files:**
- `ratings.csv`: User ratings for movies
- `movies.csv`: Movie information (title, genres)
- `links.csv`: Links to IMDb and TMDb
- `tags.csv`: User-generated tags

> **Citation**: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.

## 🧮 Implemented Algorithms

### 1. Slope One Algorithm
A simple yet effective item-based collaborative filtering algorithm that predicts ratings based on the average difference between items rated by users.

**Key Features:**
- Simple mathematical structure (uses `x + b` formula)
- Computationally efficient
- Lower risk of overfitting
- Best for systems with many users rating similar items

### 2. k-Nearest Neighbors (kNN) with Z-Score
A neighborhood-based algorithm that finds similar users/items and predicts ratings based on their preferences.

**Similarity Measures Implemented:**
- **Cosine Similarity**: Measures angle between rating vectors
- **Pearson Correlation**: Accounts for user rating bias
- **Mean Squared Difference (MSD)**: Based on rating differences

**Configuration:**
- User-based collaborative filtering
- Z-Score normalization for rating adjustments
- Optimal k = 38 (determined via grid search)

### 3. Singular Value Decomposition (SVD)
A matrix factorization technique that decomposes the user-item matrix into latent factor representations.

**Formula:**
```
r̂_ui = μ + b_u + b_i + q_i^T · p_u
```

**Parameters (optimized via grid search):**
- `n_factors`: 17
- `n_epochs`: 30
- `reg_all`: 0.04

### 4. SVD++
An enhanced version of SVD that incorporates implicit feedback in addition to explicit ratings.

**Key Enhancement:**
- Uses auxiliary feature vectors to capture implicit user-item interactions
- Better performance when users have more implicit interactions than explicit ratings

**Parameters:**
- `n_factors`: 20
- `n_epochs`: 30
- `reg_all`: 0.05

## 📈 Evaluation Metrics

### Rating Prediction Accuracy
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more heavily
- **MAE (Mean Absolute Error)**: Linear penalty for errors

### Ranking Metrics (at k=20, threshold=3.5)
- **Precision@k**: Fraction of recommended items that are relevant
- **Recall@k**: Fraction of relevant items that are recommended
- **F1-Score@k**: Harmonic mean of precision and recall

### Cross-Validation
- **10-Fold Cross-Validation**: Used for robust evaluation
- **Grid Search**: Hyperparameter optimization with cross-validation

## 🏗️ Project Structure

```
movielens-recommender/
├── api/                        # REST API
│   ├── app.py                  # FastAPI application
│   └── model.py                # Recommendation utilities
├── core/                       # Core algorithms
│   ├── utils.py                # Data loading & model persistence
│   ├── evals.py                # Evaluation metrics
│   ├── train_*.py              # Training scripts
│   ├── test_*.py               # Testing scripts
│   └── gs_*.py                 # Grid search scripts
├── data/
│   └── ml-latest-small/        # MovieLens dataset
├── docs/                       # Documentation
│   ├── introduction/           # Project introduction
│   ├── Baseline/               # Baseline predictors
│   ├── kNN-based algorithm/    # kNN documentation
│   ├── SVD algorithm/          # SVD documentation
│   ├── SVD++/                  # SVD++ documentation
│   ├── Slope One/              # Slope One documentation
│   └── metrics/                # Evaluation metrics
├── notebooks/                  # Jupyter notebooks
│   ├── movielens_visualization.ipynb
│   ├── knn_zscore.ipynb
│   ├── svd.ipynb
│   ├── svdpp.ipynb
│   ├── slope_one.ipynb
│   └── summary.ipynb
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Pinminh/movielens-recommender.git
   cd movielens-recommender
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Training Models

```bash
# Train all models
cd core
python train_slope_one.py
python train_knn_zscore.py
python train_svd.py
python train_svdpp.py
```

### Running Evaluations

```bash
# Cross-validation for each model
python test_slope_one.py
python test_knn_zscore.py
python test_svd.py
python test_svdpp.py
```

### Grid Search (Hyperparameter Tuning)

```bash
# Find optimal hyperparameters
python gs_knn_zscore_pearson.py
python gs_svd_factors.py
python gs_svdpp_factors.py
```

### Starting the API

```bash
cd api
uvicorn app:app --reload
```

## 🌐 API Endpoints

### Predict Rating
```http
GET /predict?uid={user_id}&iid={movie_id}&model={model_name}
```
- `uid`: User ID
- `iid`: Movie ID
- `model`: `slope_one` or `knn_zscore`

**Example Response:**
```json
{
  "uid": 1,
  "iid": 50,
  "est_r": 4.2,
  "details": {...}
}
```

### Get Recommendations
```http
GET /recommend?uid={user_id}&k={num_items}&model={model_name}
```
- `uid`: User ID
- `k`: Number of recommendations (default: 10)
- `model`: `slope_one` or `knn_zscore`

**Example Response:**
```json
{
  "uid": 1,
  "top_k": [
    {"movieId": "318", "est_r": 4.8, "title": "Shawshank Redemption, The (1994)", ...},
    ...
  ]
}
```

### Get Similar Users/Items
```http
GET /neighbors?uid={user_id}&k={num_neighbors}&model={model_name}
```
- `uid`: User (or Item) ID
- `k`: Number of neighbors (default: 10)
- `model`: Must be a kNN-based model

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `movielens_visualization.ipynb` | Dataset exploration and visualization |
| `knn_zscore.ipynb` | kNN algorithm analysis and experiments |
| `svd.ipynb` | SVD algorithm implementation and tuning |
| `svdpp.ipynb` | SVD++ algorithm analysis |
| `slope_one.ipynb` | Slope One algorithm demonstration |
| `summary.ipynb` | Comparative analysis of all algorithms |

## 📚 Documentation

Detailed algorithm explanations and mathematical formulations are available in the `docs/` folder:

- **Introduction**: Overview of recommender systems and collaborative filtering
- **Baseline**: User and item bias modeling
- **kNN-based Algorithm**: Similarity measures and prediction methods
- **SVD Algorithm**: Matrix factorization approach
- **SVD++**: Extended SVD with implicit feedback
- **Slope One**: Simple item-based prediction
- **Metrics**: RMSE, MAE, Precision, Recall, F1-Score

## 🔧 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| scikit-surprise | 1.1.4 | Recommendation algorithms |
| pandas | 2.2.3 | Data manipulation |
| numpy | 1.26.4 | Numerical computations |
| matplotlib | 3.9.2 | Visualization |
| seaborn | 0.13.2 | Statistical visualization |
| fastapi | 0.115.3 | REST API framework |
| uvicorn | 0.32.0 | ASGI server |

## 📄 License

This project uses the MovieLens dataset, which is subject to the following conditions:
- Acknowledgment required in publications
- No commercial use without permission from GroupLens Research
- Redistribution allowed under the same conditions

## 🙏 Acknowledgments

- **Ho Chi Minh City University of Technology (HCMUT)** - Academic guidance
- **GroupLens Research** - MovieLens dataset
- **Surprise Library** - Recommendation algorithm implementations

## 📬 Contact

For questions or collaboration, please open an issue on the repository.

---

*This project is for educational purposes as part of HCMUT's Computer Science curriculum.*