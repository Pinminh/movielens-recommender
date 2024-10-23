import numpy as np

from surprise import KNNWithZScore
from surprise.model_selection import GridSearchCV
from utils import load_movielens, seed_random_state, store_gscv


gs_name = "gs_knn_zscore"
data = load_movielens()

param_grid = {
    "k": np.arange(10, 51, 1),
    "sim_options": {
        "name": ["msd", "cosine", "pearson"],
        "user_based": [False, True],
    }
}

seed_random_state()
gs = GridSearchCV(KNNWithZScore, param_grid, measures=["rmse", "mae"], cv=10, joblib_verbose=3, n_jobs=2)
gs.fit(data)

store_gscv(gs_name, gs)